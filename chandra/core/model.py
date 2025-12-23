"""Model inference and schema definitions."""
from typing import List
from dataclasses import dataclass
import base64
import io
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat

from PIL import Image
from openai import OpenAI
import torch
import gc

from chandra.settings import settings
from chandra.core.output import (
    parse_markdown,
    parse_html,
    parse_chunks,
    extract_images,
    scale_to_fit,
    detect_repeat_token,
)

from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor, StoppingCriteria, StoppingCriteriaList



class StopAfterJSONFence(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.text_buffer = ""

        self.in_fence = False
        self.json_started = False
        self.brace_depth = 0
        self.json_finished = False

    def __call__(self, input_ids, scores, **kwargs):
        token = self.tokenizer.decode(
            input_ids[0][-1:], skip_special_tokens=True
        )
        self.text_buffer += token

        # ---- Detect fence ```
        if "```" in self.text_buffer[-10:]:
            if not self.in_fence:
                # fence mở
                self.in_fence = True
                return False
            else:
                # fence đóng
                if self.json_finished:
                    return True  # <<< STOP ĐÚNG LÚC
                return False

        # ---- Track JSON braces ONLY inside fence
        if self.in_fence:
            for ch in token:
                if ch == "{":
                    self.brace_depth += 1
                    self.json_started = True
                elif ch == "}":
                    self.brace_depth -= 1

            if self.json_started and self.brace_depth == 0:
                self.json_finished = True

        return False

# ============= Schema Definitions =============

@dataclass
class GenerationResult:
    """Result from model generation."""
    raw: str
    token_count: int
    error: bool = False


@dataclass
class BatchInputItem:
    """Input item for batch inference."""
    image: Image.Image
    prompt: str | None = None
    prompt_type: str | None = None


@dataclass
class BatchOutputItem:
    """Output item from batch inference."""
    markdown: str
    html: str
    chunks: dict
    raw: str
    page_box: List[int]
    token_count: int
    images: dict
    error: bool

def is_json_prompt(prompt: str | None) -> bool:
    return prompt is not None and "STRICT JSON" in prompt

import json
def extract_strict_json(text: str) -> str | None:
    end = text.find("END_JSON")
    if end != -1:
        text = text[:end]

def cut_at_assistant(text: str) -> str:
    """
    Remove everything after assistant role leakage.
    """
    markers = [
        "\nassistant",
        "\nAssistant",
        "<|assistant|>",
    ]
    for m in markers:
        idx = text.find(m)
        if idx != -1:
            return text[:idx].strip()
    return text.strip()


def extract_first_json(text: str) -> str | None:
    """
    Extract the first valid JSON object from model output.
    """
    text = cut_at_assistant(text)

    # Remove markdown fences if any
    text = text.replace("```json", "").replace("```", "").strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return None

    candidate = text[start:end + 1]

    try:
        json.loads(candidate)
        return candidate
    except json.JSONDecodeError:
        return None


def limit_blocks(json_text: str, max_blocks: int = 50) -> str:
    data = json.loads(json_text)
    if "blocks" in data and isinstance(data["blocks"], list):
        data["blocks"] = data["blocks"][:max_blocks]
    return json.dumps(data, ensure_ascii=False)


# ============= Inference Manager =============

class InferenceManager:
    """Manager for running inference."""
    
    def __init__(self, method: str = "Qwen3"):
        assert method in ("Qwen3", "Paddle", "VLLM"), "method must be one of 'Qwen3', 'Paddle', or 'VLLM'"
        self.method = method

        if method == "Qwen3":
            self.model = load_qwen()
        elif method == "Paddle":
            self.model = None  # Placeholder for Paddle model loading
        else:
            self.model = None

    def generate(
        self, batch: List[BatchInputItem], max_output_tokens=None, **kwargs
    ) -> List[BatchOutputItem]:
        output_kwargs = {}
        if "include_images" in kwargs:
            output_kwargs["include_images"] = kwargs.pop("include_images")
        if "include_headers_footers" in kwargs:
            output_kwargs["include_headers_footers"] = kwargs.pop(
                "include_headers_footers"
            )
        bbox_scale = kwargs.pop("bbox_scale", settings.BBOX_SCALE)
        vllm_api_base = kwargs.pop("vllm_api_base", settings.VLLM_API_BASE)

        if self.method == "vllm":
            results = generate_vllm(
                batch,
                max_output_tokens=max_output_tokens,
                bbox_scale=bbox_scale,
                vllm_api_base=vllm_api_base,
                **kwargs,
            )
        else:
            results = generate_hf(
                batch,
                self.model,
                max_output_tokens=max_output_tokens,
                bbox_scale=bbox_scale,
                **kwargs,
            )

        output = []
        for result, input_item in zip(results, batch):
            chunks = parse_chunks(result.raw, input_item.image, bbox_scale=bbox_scale)
            output.append(
                BatchOutputItem(
                    markdown=parse_markdown(result.raw, **output_kwargs),
                    html=parse_html(result.raw, **output_kwargs),
                    chunks=chunks,
                    raw=result.raw,
                    page_box=[0, 0, input_item.image.width, input_item.image.height],
                    token_count=result.token_count,
                    images=extract_images(result.raw, chunks, input_item.image),
                    error=result.error,
                )
            )
        return output


# ============= HuggingFace Backend =============

def load_qwen():
    """Load Qwen3-VL-2B-Instruct model from HuggingFace."""
    
    MODEL_CHECKPOINT = "Qwen/Qwen3-VL-2B-Instruct"

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_CHECKPOINT,
        dtype=torch.float16,
        device_map="cuda:0",
        attn_implementation="sdpa"
    ).eval()

    processor = Qwen3VLProcessor.from_pretrained(MODEL_CHECKPOINT)
    model.processor = processor
    return model


def generate_hf(
    batch: List[BatchInputItem],
    model,
    max_output_tokens=None,
    bbox_scale: int = settings.BBOX_SCALE,
    **kwargs,
) -> List[GenerationResult]:

    from chandra.settings import PROMPTS
    from qwen_vl_utils import process_vision_info

    if max_output_tokens is None:
        max_output_tokens = settings.MAX_OUTPUT_TOKENS

    messages = []

    for item in batch:
        prompt = item.prompt
        if not prompt:
            prompt = PROMPTS[item.prompt_type].replace(
                "{bbox_scale}", str(bbox_scale)
            )

        image = scale_to_fit(item.image)

        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]
        })

    # 🔥 BẮT BUỘC: chat template
    text = model.processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, _ = process_vision_info(messages)

    inputs = model.processor(
        text=text,
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.inference_mode():
        stopping_criteria = StoppingCriteriaList([StopAfterJSONFence(model.processor.tokenizer)])
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=min(max_output_tokens, 2048),
            do_sample=False,
            # eos_token_id=model.config.eos_token_id,
            stopping_criteria = stopping_criteria
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    outputs = model.processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    results = []
    for text in outputs:
        text = cut_at_assistant(text)

        end = text.rfind("}")
        if end != -1:
            text = text[:end + 1]

        results.append(
            GenerationResult(
                raw=text.strip(),
                token_count=len(text),
                error=False,
            )
        )

    return results



def _process_batch_element_hf(item: BatchInputItem, processor, bbox_scale: int):
    """Process a single batch element for HF model."""
    from chandra.settings import PROMPTS
    
    prompt = item.prompt
    prompt_type = item.prompt_type

    if not prompt:
        prompt = PROMPTS[prompt_type].replace("{bbox_scale}", str(bbox_scale))

    content = []
    image = scale_to_fit(item.image)
    content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})
    message = {"role": "user", "content": content}
    return message


# ============= vLLM Backend =============

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def generate_vllm(
    batch: List[BatchInputItem],
    max_output_tokens: int = None,
    max_retries: int = None,
    max_workers: int | None = None,
    custom_headers: dict | None = None,
    max_failure_retries: int | None = None,
    bbox_scale: int = settings.BBOX_SCALE,
    vllm_api_base: str = settings.VLLM_API_BASE,
) -> List[GenerationResult]:
    """Generate with vLLM server."""
    from chandra.settings import PROMPTS
    
    client = OpenAI(
        api_key=settings.VLLM_API_KEY,
        base_url=vllm_api_base,
        default_headers=custom_headers,
    )
    model_name = settings.VLLM_MODEL_NAME

    if max_retries is None:
        max_retries = settings.MAX_VLLM_RETRIES

    if max_workers is None:
        max_workers = min(64, len(batch))

    if max_output_tokens is None:
        max_output_tokens = settings.MAX_OUTPUT_TOKENS

    if model_name is None:
        models = client.models.list()
        model_name = models.data[0].id

    def _generate(
        item: BatchInputItem, temperature: float = 0, top_p: float = 0.1
    ) -> GenerationResult:
        prompt = item.prompt
        if not prompt:
            prompt = PROMPTS[item.prompt_type].replace(
                "{bbox_scale}", str(bbox_scale)
            )

        content = []
        image = scale_to_fit(item.image)
        image_b64 = image_to_base64(image)
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
            }
        )

        content.append({"type": "text", "text": prompt})

        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": content}],
                max_tokens=max_output_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            raw = completion.choices[0].message.content
            result = GenerationResult(
                raw=raw,
                token_count=completion.usage.completion_tokens,
                error=False,
            )
        except Exception as e:
            print(f"Error during VLLM generation: {e}")
            return GenerationResult(raw="", token_count=0, error=True)

        return result

    def _should_retry(result, retries, max_retries, max_failure_retries=None):
        if result.error:
            if max_failure_retries is None:
                return retries < max_retries
            else:
                return retries < max_failure_retries
        return detect_repeat_token(result.raw) and retries < max_retries

    def process_item(item, max_retries, max_failure_retries=None):
        result = _generate(item)
        retries = 0

        while _should_retry(result, retries, max_retries, max_failure_retries):
            result = _generate(item, temperature=0.3, top_p=0.95)
            retries += 1
            time.sleep(0.1)

        return result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            executor.map(
                process_item,
                batch,
                repeat(max_retries),
                repeat(max_failure_retries),
            )
        )

    return results


def cleanup_memory():
    """Clean up GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()

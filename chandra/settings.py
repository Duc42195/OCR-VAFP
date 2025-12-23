"""Settings and configuration."""
from dotenv import find_dotenv
from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
    """Global settings configuration."""
    # Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    IMAGE_DPI: int = 192
    MIN_PDF_IMAGE_DIM: int = 1024
    MIN_IMAGE_DIM: int = 1536
    MODEL_CHECKPOINT: str = "datalab-to/chandra"
    TORCH_DEVICE: str | None = None
    MAX_OUTPUT_TOKENS: int = 3072
    TORCH_ATTN: str | None = None
    BBOX_SCALE: int = 1024

    # vLLM server settings
    VLLM_API_KEY: str = "K86318648388957"
    VLLM_API_BASE: str = "https://api.ocr.space/parse/image"
    VLLM_MODEL_NAME: str = "OCR Space"
    VLLM_GPUS: str = "0"
    MAX_VLLM_RETRIES: int = 6

    class Config:
        env_file = find_dotenv("local.env")
        extra = "ignore"


settings = Settings()


# ============= Prompt Templates =============

ALLOWED_TAGS = [
    "math",
    "br",
    "i",
    "b",
    "u",
    "del",
    "sup",
    "sub",
    "table",
    "tr",
    "td",
    "p",
    "th",
    "div",
    "pre",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "ul",
    "ol",
    "li",
    "input",
    "a",
    "span",
    "img",
    "hr",
    "tbody",
    "small",
    "caption",
    "strong",
    "thead",
    "big",
    "code",
]

ALLOWED_ATTRIBUTES = [
    "class",
    "colspan",
    "rowspan",
    "display",
    "checked",
    "type",
    "border",
    "value",
    "style",
    "href",
    "alt",
    "align",
]

PROMPT_ENDING = f"""
Only use these tags {ALLOWED_TAGS}, and these attributes {ALLOWED_ATTRIBUTES}.

Guidelines:
* Inline math: Surround math with <math>...</math> tags. Math expressions should be rendered in KaTeX-compatible LaTeX. Use display for block math.
* Tables: Use colspan and rowspan attributes to match table structure.
* Formatting: Maintain consistent formatting with the image, including spacing, indentation, subscripts/superscripts, and special characters.
* Images: Include a description of any images in the alt attribute of an <img> tag. Do not fill out the src property.
* Forms: Mark checkboxes and radio buttons properly.
* Text: join lines together properly into paragraphs using <p>...</p> tags.  Use <br> tags for line breaks within paragraphs, but only when absolutely necessary to maintain meaning.
* Use the simplest possible HTML structure that accurately represents the content of the block.
* Make sure the text is accurate and easy for a human to read and interpret.  Reading order should be correct and natural.
""".strip()

OCR_LAYOUT_PROMPT = f"""
OCR this image to HTML, arranged as layout blocks.  Each layout block should be a div with the data-bbox attribute representing the bounding box of the block in [x0, y0, x1, y1] format.  Bboxes are normalized 0-{{bbox_scale}}. The data-label attribute is the label for the block.

Use the following labels:
- Caption
- Footnote
- Equation-Block
- List-Group
- Page-Header
- Page-Footer
- Image
- Section-Header
- Table
- Text
- Complex-Block
- Code-Block
- Form
- Table-Of-Contents
- Figure
End the output immediately after the last layout block.
Do NOT add any content after that.
{PROMPT_ENDING}
""".strip()

OCR_PROMPT = f"""
OCR this image to HTML.

{PROMPT_ENDING}
""".strip()
OCR_JSON_PROMPT = """
You are an OCR system.

TASK:
Extract all visible content from the image and output STRICT JSON.

RULES (MANDATORY):
- Output ONLY valid JSON
- No markdown
- No explanations
- No trailing text
- Stop immediately after the final }

IMPORTANT:
- EACH detected table MUST be a SEPARATE object in the "content" array
- If there are multiple tables on the page, output multiple table objects
- Do NOT merge different tables into one
- Preserve table order from top to bottom

JSON SCHEMA:
{
  "page": {
    "title": string | null,
    "subtitle": string | null,
    "content": [
      {
        "type": "table",
        "rows": [
        ["string", "string", "..."]
        ]
      }
    ]
  }
}

TABLE RULES:
- rows is a 2D array of strings
- Empty cell → empty string ""
- Header rows are INCLUDED as normal rows
- No bbox, no coordinates, no text blocks

END_JSON
""".strip()

PROMPTS = {
    "ocr_layout": OCR_LAYOUT_PROMPT,
    "ocr": OCR_PROMPT,
    "ocr_json": OCR_JSON_PROMPT,
}

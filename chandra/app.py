# """Streamlit web application for Chandra OCR."""
import pypdfium2 as pdfium
import streamlit as st
from PIL import Image
import base64
from io import BytesIO
import re
import time
import hashlib

from chandra.core.model import BatchInputItem, InferenceManager, cleanup_memory
from chandra.core.output import (
    draw_layout, parse_layout, extract_first_fenced_json_autofix, 
    save_json, extract_tables_from_json, table_block_to_df, df_to_table_block, 
    export_from_json, extract_metadata_from_json, update_metadata_in_json
)
from chandra.core.input import load_pdf_images


@st.cache_resource()
def load_model(method: str):
    """Load model with caching."""
    return InferenceManager(method=method)


@st.cache_data()
def get_page_image(pdf_file, page_num):
    """Get a specific page from PDF with caching."""
    return load_pdf_images(pdf_file, [page_num])[0]


@st.cache_data()
def page_counter(pdf_file):
    """Count pages in PDF with caching."""
    doc = pdfium.PdfDocument(pdf_file)
    doc_len = len(doc)
    doc.close()
    return doc_len


def pil_image_to_base64(pil_image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL image to base64 data URL."""
    buffered = BytesIO()
    pil_image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/{format.lower()};base64,{img_str}"


def embed_images_in_markdown(markdown: str, images: dict) -> str:
    """Replace image filenames in markdown with base64 data URLs."""
    for img_name, pil_image in images.items():
        data_url = pil_image_to_base64(pil_image, format="PNG")
        pattern = rf'(!\[.*?\])\({re.escape(img_name)}(?:\s+"[^"]*")?\)'
        markdown = re.sub(pattern, rf"\1({data_url})", markdown)
    return markdown


def ocr_layout(
    img: Image.Image,
    model=None,
):
    """Run OCR with layout detection."""
    batch = BatchInputItem(
        image=img,
        prompt_type="ocr_json",
    )
    result = model.generate([batch])[0]
    layout = parse_layout(result.raw, img)
    layout_image = draw_layout(img, layout)
    return result, layout_image


# ============= Streamlit App =============

st.set_page_config(layout="wide", page_title="Chandra OCR Demo")

st.markdown("""
# Chandra OCR Demo

This app lets you try Chandra, a layout-aware vision language model for: **OCR with Layout Detection**: Understand document structure
""")

# ============= Initialize ALL Session State =============
if "table_dfs" not in st.session_state:
    st.session_state.table_dfs = {}

if "tables" not in st.session_state:
    st.session_state.tables = None

if "json_data" not in st.session_state:
    st.session_state.json_data = None

if "layout_image" not in st.session_state:
    st.session_state.layout_image = None

if "result" not in st.session_state:
    st.session_state.result = None

if "elapse_time" not in st.session_state:
    st.session_state.elapse_time = 0

if "current_file_id" not in st.session_state:
    st.session_state.current_file_id = None

if "raw_output" not in st.session_state:
    st.session_state.raw_output = None

if "parse_error" not in st.session_state:
    st.session_state.parse_error = None

# Get model mode selection
model_mode = st.sidebar.selectbox(
    "Model Mode",
    ["None", "Qwen3", "Paddle"],
    index=1,
    help="Qwen 3 VL 2B Instruct is default, PaddleVL and PPStructure is coming soon",
)

# Only load model if a mode is selected
model = None
if model_mode == "None":
    st.warning("Please select a model mode (Qwen3 or Paddle) to run OCR.")
else:
    model = load_model(model_mode)

clear_memory = st.sidebar.button("Clear Cache", key="clear_cache_button")
if clear_memory:
    cleanup_memory()
    st.success("Cache cleared!")


# ============= Task: OCR with Layout =============
in_file = st.file_uploader(
    "PDF file or image:", type=["pdf", "png", "jpg", "jpeg", "gif", "webp"]
)

if in_file is None:
    st.stop()

# Generate unique ID for current file to detect when user uploads new file
file_id = hashlib.md5(in_file.read()).hexdigest()
in_file.seek(0)  # Reset file pointer

# Reset state when new file is uploaded
if st.session_state.current_file_id != file_id:
    st.session_state.current_file_id = file_id
    st.session_state.json_data = None
    st.session_state.tables = None
    st.session_state.table_dfs = {}
    st.session_state.layout_image = None
    st.session_state.result = None
    st.session_state.raw_output = None
    st.session_state.parse_error = None

filetype = in_file.type

col1, col2 = st.columns([0.5, 0.5])

# ============= COLUMN 1: INPUT IMAGE =============
with col1:
    st.subheader("Input Image")
    
    # Handle PDF or Image
    page_count = None
    if "pdf" in filetype:
        page_count = page_counter(in_file)
        page_number = st.slider(
            f"Page number out of {page_count}:", min_value=1, value=1, max_value=page_count
        )
        pil_image = get_page_image(in_file, page_number)
    else:
        pil_image = Image.open(in_file).convert("RGB")
        page_number = None

    if pil_image is None:
        st.stop()
    
    st.image(pil_image, width='content')

    # Run OCR Button
    run_ocr = st.button("Run OCR", key="ocr_button", use_container_width=True)

    if run_ocr:
        if model_mode == "None":
            st.error("Please select a model mode to run OCR.")
        else:
            with st.spinner("Running OCR..."):
                start_time = time.time()
                result, layout_image = ocr_layout(pil_image, model)
                cleanup_memory()
                elapse_time = time.time() - start_time
                
                # Store in session state
                st.session_state.result = result
                st.session_state.layout_image = layout_image
                st.session_state.elapse_time = elapse_time
                st.session_state.raw_output = result.raw  # Always save raw output
                
                try:
                    json_data = extract_first_fenced_json_autofix(result.raw)
                    save_json(json_data)
                    st.session_state.json_data = json_data
                    st.session_state.tables = extract_tables_from_json(json_data)
                    st.session_state.table_dfs = {}  # Reset table dfs
                    st.session_state.parse_error = None  # Clear previous errors
                    st.success("OCR completed!")
                except ValueError as e:
                    error_msg = f"Error parsing JSON: {str(e)}"
                    st.session_state.parse_error = error_msg
                    st.error(error_msg)


# ============= COLUMN 2: RESULTS =============
with col2:
    st.subheader("Results")
    
    # Create tabs for Result and Insights (always visible)
    result_tab, insights_tab = st.tabs(["Result", "Insights"])
    
    if st.session_state.json_data is None and st.session_state.raw_output is None:
        with result_tab:
            st.info("Run OCR to see results")
        with insights_tab:
            st.info("Run OCR to see insights")
    else:
        
        with result_tab:
            # Show layout image
            if st.session_state.layout_image is not None:
                st.image(st.session_state.layout_image, caption="Detected Layout", width="content")
            
            st.divider()
            
            # Edit metadata section
            st.subheader("Edit Document Metadata")
            
            # Extract metadata
            metadata = extract_metadata_from_json(st.session_state.json_data)
            
            # Create columns for metadata editing
            new_title = st.text_input(
                    "Title",
                    value=metadata.get("title", ""),
                    key="title_input"
                )
            
            new_subtitle = st.text_input(
                    "Subtitle",
                    value=metadata.get("subtitle", ""),
                    key="subtitle_input"
                )
            
            # Update metadata in session state
            if new_title != metadata.get("title", "") or new_subtitle != metadata.get("subtitle", ""):
                st.session_state.json_data = update_metadata_in_json(
                    st.session_state.json_data,
                    {"title": new_title, "subtitle": new_subtitle}
                )
            
            st.divider()
            st.subheader("Edit Tables")
            
            # Edit tables section
            tables = st.session_state.tables

            if not tables:
                st.warning("No table is found in JSON")
            else:
                for idx, table_item in enumerate(tables):
                    # Handle both dict with "type" and wrapped dict with "data"
                    if isinstance(table_item, dict) and "type" in table_item:
                        table = table_item
                    elif isinstance(table_item, dict) and "data" in table_item:
                        table = table_item.get("data", table_item)
                    else:
                        table = table_item
                    
                    st.subheader(f"Table {idx + 1}")

                    # Initialize dataframe if not exists
                    if idx not in st.session_state.table_dfs:
                        st.session_state.table_dfs[idx] = table_block_to_df(table)
                    
                    # Get current dataframe from state
                    current_df = st.session_state.table_dfs[idx]
                    
                    # Data editor with unique key
                    editted_df = st.data_editor(
                        current_df,
                        key=f"table_{idx}",
                        width="stretch",
                        num_rows="dynamic",
                    )
                    
                    # Update state immediately after edit
                    st.session_state.table_dfs[idx] = editted_df
                    # Update the table block in tables
                    df_to_table_block(editted_df, table)
            
            # Prepare data for export
            if st.session_state.tables:
                for idx, table_item in enumerate(st.session_state.tables):
                    if isinstance(table_item, dict) and "type" in table_item:
                        table = table_item
                    elif isinstance(table_item, dict) and "data" in table_item:
                        table = table_item.get("data", table_item)
                    else:
                        table = table_item
                    
                    if idx in st.session_state.table_dfs:
                        df_to_table_block(
                            st.session_state.table_dfs[idx],
                            table
                        )
            
            # Export section
            st.divider()
            st.subheader("Download")
            
            format_choice = st.selectbox("Select a format to download", ["json", "xlsx", "docx"], index=0, key="format_select")
            
            file_bytes, mime, filename = export_from_json(
                st.session_state.json_data,
                format_choice,
                f"{in_file.name.rsplit('.', 1)[0]}_page{page_number if page_number is not None else 0}"
            )
            st.download_button(
                label="Download",
                data=file_bytes,
                file_name=filename,
                mime=mime,
                use_container_width=True
            )
        
        with insights_tab:
            # Always show performance metrics
            col_time, col_tokens, col_speed = st.columns(3)
            
            with col_time:
                st.metric("Elapsed Time (s)", f"{st.session_state.elapse_time:.2f}")
            
            if st.session_state.result is not None:
                with col_tokens:
                    st.metric("Tokens Used (tk)", f"{st.session_state.result.token_count}")
                
                with col_speed:
                    speed = st.session_state.result.token_count / st.session_state.elapse_time if st.session_state.elapse_time > 0 else 0
                    st.metric("Speed (tk/s)", f"{speed:.2f}")
                
                st.divider()
                st.subheader("Full Output")
                st.text(str(st.session_state.result))
            else:
                with col_tokens:
                    st.metric("Tokens Used (tk)", "N/A")
                
                with col_speed:
                    st.metric("Speed (tk/s)", "N/A")
                
                st.divider()
            
            # Show parse error if exists
            if st.session_state.parse_error:
                st.error(f"🔴 {st.session_state.parse_error}")
                st.subheader("Raw Output (for debugging)")
                if st.session_state.raw_output:
                    st.text_area("Raw OCR Output", st.session_state.raw_output, height=300, disabled=True)
            elif st.session_state.raw_output and st.session_state.json_data is None:
                st.warning("⚠️ OCR completed but JSON parsing failed. Check raw output below.")
                st.subheader("Raw Output (for debugging)")
                st.text_area("Raw OCR Output", st.session_state.raw_output, height=300, disabled=True)
            elif not st.session_state.raw_output:
                st.warning("⚠️ OCR inference did not complete or no output available. This may happen with very large files.")

cleanup_memory()
# My Chandra

Chandra is a highly accurate OCR model that converts images and PDFs into structured HTML/Markdown/JSON while preserving layout information.

## Key Differences
- Preserves layout: tables, forms, mathematical formulas, and images.

- Supports handwriting and multiple languages ​​(40+).

- Outputs Markdown/HTML/JSON files with metadata and images.

## Quick Installation & Run
```bash
pip install chandra-ocr

# CLI
chandra input.pdf ./output --method hf # local
chandra input.pdf ./output --method vllm # server

# Streamlit demo
chandra_app

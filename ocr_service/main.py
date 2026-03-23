"""
ocr_service/main.py
-------------------
Lightweight FastAPI microservice that exposes a single /ocr endpoint.

Accepts a PDF or image file and returns its text content extracted via
Tesseract OCR. PDFs are first rasterised to images with pdf2image (which
wraps Poppler's pdftoppm) at 300 DPI to ensure good OCR quality.
"""

import os

import pytesseract
from fastapi import FastAPI, File, UploadFile
from pdf2image import convert_from_path
from PIL import Image

# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OCR Service",
    description="Extracts text from scanned PDFs and images via Tesseract.",
    version="1.0.0",
)

# Magic bytes that identify PDF files.
# Checking the file header is more reliable than relying on the filename
# extension, because the client might send a file without an extension.
PDF_MAGIC_BYTES = b"%PDF"

# DPI used when rasterising PDF pages.
# 300 DPI is the standard recommendation for high-quality OCR.
OCR_DPI = 300


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@app.post("/ocr", summary="Extract text from a PDF or image via OCR")
async def ocr(file: UploadFile = File(...)):
    """
    Perform OCR on an uploaded file and return the extracted text.

    The endpoint auto-detects the file type by inspecting the first four bytes:
    - If they match ``%PDF``, the file is treated as a PDF and each page is
      rasterised before being passed to Tesseract.
    - Otherwise, the file is opened directly as an image with Pillow.

    Args:
        file: Uploaded file (PDF or common image format: PNG, JPEG, TIFF …).

    Returns:
        {"text": "<extracted text>"}

    Notes:
        The temporary file written to disk is always deleted in the ``finally``
        block, even if an exception occurs during processing.
    """
    temp_path = "temp_file"
    content = await file.read()

    # Write to disk: pdf2image and Tesseract both require a file path,
    # not an in-memory byte stream.
    with open(temp_path, "wb") as f:
        f.write(content)

    text = ""

    try:
        if content[:4] == PDF_MAGIC_BYTES:
            # --- PDF path: rasterise each page then run OCR ---
            images = convert_from_path(temp_path, dpi=OCR_DPI)
            for img in images:
                text += pytesseract.image_to_string(img)
        else:
            # --- Image path: open directly and run OCR ---
            image = Image.open(temp_path)
            text = pytesseract.image_to_string(image)

        return {"text": text}

    finally:
        # Guarantee cleanup regardless of success or failure
        if os.path.exists(temp_path):
            os.remove(temp_path)

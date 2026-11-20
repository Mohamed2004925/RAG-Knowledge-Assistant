"""Small, inspectable file-to-text functions for Chapter 2."""

from io import BytesIO
from pathlib import Path

from pypdf import PdfReader


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def extract_document(filename: str, file_bytes: bytes) -> list[dict[str, str]]:
    """Extract text sections while keeping simple page/section labels as metadata."""
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(BytesIO(file_bytes), strict=False)
        if reader.is_encrypted:
            raise ValueError("This PDF is encrypted. Provide an unlocked copy for this demo.")

        return [
            {"label": f"Page {page_number}", "text": page.extract_text() or ""}
            for page_number, page in enumerate(reader.pages, start=1)
        ]

    if suffix in {".txt", ".md"}:
        text = file_bytes.decode("utf-8-sig", errors="replace")
        return [{"label": "Document text", "text": text}]

    if suffix in IMAGE_EXTENSIONS:
        # Keep photos in the same page-based pipeline. OCR is applied by the UI
        # before indexing; without OCR this remains intentionally empty.
        return [{"label": "Page 1", "text": ""}]

    raise ValueError(f"Unsupported file type: {suffix or 'no extension'}")

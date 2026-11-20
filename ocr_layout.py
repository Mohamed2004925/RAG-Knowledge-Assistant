"""Optional YOLO layout detection and Tesseract OCR for scanned PDF pages."""

from __future__ import annotations

import os
import shutil
from io import BytesIO
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

DEFAULT_WEIGHTS = Path(__file__).parent / "best_model" / "doclaynet_yolov8s_baseline.pt"
TEXT_CLASSES = {
    "Text",
    "Title",
    "Section-header",
    "List-item",
    "Caption",
    "Footnote",
    "Page-header",
    "Page-footer",
}
OCR_CLASSES = TEXT_CLASSES | {"Table"}
BASE_UPSCALE = 2
MARGIN = 10


def configure_tesseract(command: str | None = None) -> str | None:
    """Find Tesseract from a configured path, PATH, or common Windows location."""
    import pytesseract

    configured = command or os.environ.get("TESSERACT_CMD")
    current = configured or pytesseract.pytesseract.tesseract_cmd

    if Path(current).is_file():
        pytesseract.pytesseract.tesseract_cmd = str(Path(current))
        return pytesseract.pytesseract.tesseract_cmd

    located = shutil.which(current) or shutil.which("tesseract")
    if located:
        pytesseract.pytesseract.tesseract_cmd = located
        return located

    for candidate in (
        Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        / "Tesseract-OCR"
        / "tesseract.exe",
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
        / "Tesseract-OCR"
        / "tesseract.exe",
    ):
        if candidate.is_file():
            pytesseract.pytesseract.tesseract_cmd = str(candidate)
            return str(candidate)
    return None


def load_model(weights_path: Path = DEFAULT_WEIGHTS):
    """Load and verify the document-layout detector's expected class labels."""
    if not weights_path.is_file():
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")

    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    names = model.names
    labels = set(names.values()) if isinstance(names, dict) else set(names)
    if "Text" not in labels:
        raise RuntimeError(
            "The checkpoint loaded, but its label map does not include the expected "
            "'Text' class. Check that best_model contains the document-layout model."
        )
    return model


def _page_upscale(image_shape: tuple[int, ...]) -> int:
    largest_side = max(image_shape[:2])
    if largest_side <= 0:
        return BASE_UPSCALE
    return min(3, max(1, round(BASE_UPSCALE * 1025 / largest_side)))


def _ocr_crop(image_bgr: Any, scale: int) -> str:
    import cv2
    import pytesseract

    height, width = image_bgr.shape[:2]
    if height == 0 or width == 0:
        return ""

    enlarged = cv2.resize(
        image_bgr,
        (width * scale, height * scale),
        interpolation=cv2.INTER_CUBIC,
    )
    padded = cv2.copyMakeBorder(
        enlarged,
        MARGIN,
        MARGIN,
        MARGIN,
        MARGIN,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
    image_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    return pytesseract.image_to_string(
        image_rgb,
        lang="eng",
        config="--oem 3 --psm 6",
        timeout=45,
    ).strip()


def _detect_and_ocr(image: Any, model: Any) -> list[dict[str, Any]]:
    import cv2

    result = model.predict(
        source=image,
        imgsz=640,
        conf=0.25,
        agnostic_nms=True,
        verbose=False,
    )[0]
    names = model.names
    height, width = image.shape[:2]
    scale = _page_upscale(image.shape)
    boxes = sorted(
        result.boxes,
        key=lambda box: (float(box.xyxy[0][1]), float(box.xyxy[0][0])),
    )
    regions = []

    for box in boxes:
        class_id = int(box.cls.item())
        label = names[class_id] if not isinstance(names, dict) else names[class_id]
        if label not in OCR_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        regions.append(
            {
                "layout_label": label,
                "layout_confidence": round(float(box.conf.item()), 4),
                "bbox_xyxy": [x1, y1, x2, y2],
                "text": _ocr_crop(image[y1:y2, x1:x2], scale),
            }
        )

    if not regions:
        regions.append(
            {
                "layout_label": "full_page_fallback",
                "layout_confidence": None,
                "bbox_xyxy": [0, 0, width, height],
                "text": _ocr_crop(image, scale),
            }
        )
    return regions


def _merge_text(primary_text: str, supplemental_text: str) -> str:
    """Preserve full-page order and append layout text not found in that pass."""
    primary_lines = [line.strip() for line in primary_text.splitlines() if line.strip()]
    merged = list(primary_lines)
    normalized_lines = [
        "".join(char for char in line.casefold() if char.isalnum())
        for line in primary_lines
    ]

    for line in (line.strip() for line in supplemental_text.splitlines() if line.strip()):
        normalized = "".join(char for char in line.casefold() if char.isalnum())
        if not normalized:
            continue
        duplicate = any(
            normalized == existing
            or (
                min(len(normalized), len(existing)) >= 6
                and SequenceMatcher(None, normalized, existing).ratio() >= 0.88
            )
            for existing in normalized_lines
        )
        if not duplicate:
            merged.append(line)
            normalized_lines.append(normalized)
    return "\n".join(merged)


def ocr_pdf_pages(
    file_bytes: bytes,
    page_numbers: list[int],
    model: Any | None = None,
    dpi: int = 150,
) -> dict[int, dict[str, Any]]:
    """OCR selected one-based PDF pages, optionally using YOLO layout guidance."""
    import cv2
    import numpy as np
    import pymupdf
    import pytesseract

    if not configure_tesseract():
        raise RuntimeError(
            "Tesseract was not found. Install it or set the TESSERACT_CMD environment variable."
        )

    results: dict[int, dict[str, Any]] = {}
    scale = dpi / 72
    with pymupdf.open(stream=file_bytes, filetype="pdf") as document:
        for page_number in page_numbers:
            page = document[page_number - 1]
            pixmap = page.get_pixmap(
                matrix=pymupdf.Matrix(scale, scale),
                alpha=False,
            )
            rgb = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                pixmap.height, pixmap.width, pixmap.n
            )
            image_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if model is None:
                regions = [
                    {
                        "layout_label": "full_page_fallback",
                        "layout_confidence": None,
                        "bbox_xyxy": [0, 0, image_bgr.shape[1], image_bgr.shape[0]],
                        "text": _ocr_crop(image_bgr, _page_upscale(image_bgr.shape)),
                    }
                ]
            else:
                regions = _detect_and_ocr(image_bgr, model)

            if len(regions) == 1 and regions[0]["layout_label"] == "full_page_fallback":
                full_page_text = regions[0]["text"]
            else:
                full_page_text = _ocr_crop(
                    image_bgr,
                    scale=_page_upscale(image_bgr.shape),
                )

            layout_text = "\n".join(region["text"] for region in regions if region["text"])
            merged_text = _merge_text(full_page_text, layout_text)
            results[page_number] = {
                "text": merged_text,
                "regions": regions,
                "text_source": "merged" if merged_text != full_page_text else "full_page",
                "preprocessing": {
                    "pdf_dpi": dpi,
                    "upscale": _page_upscale(image_bgr.shape),
                    "white_margin_px": MARGIN,
                    "psm": 6,
                },
            }
    return results


def ocr_image(
    file_bytes: bytes,
    model: Any | None = None,
) -> dict[str, Any]:
    """Recognize text from one photo or scanned image using local Tesseract OCR."""
    import cv2
    import numpy as np
    import pytesseract
    from PIL import Image, ImageOps

    if not configure_tesseract():
        raise RuntimeError(
            "Tesseract was not found. Install it or set the TESSERACT_CMD environment variable."
        )

    try:
        with Image.open(BytesIO(file_bytes)) as image:
            oriented = ImageOps.exif_transpose(image).convert("RGB")
            image_rgb = np.asarray(oriented)
    except Exception as error:
        raise ValueError("This image could not be opened. Try a JPG or PNG photo.") from error

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if model is None:
        regions = [
            {
                "layout_label": "full_image",
                "layout_confidence": None,
                "bbox_xyxy": [0, 0, image_bgr.shape[1], image_bgr.shape[0]],
                "text": _ocr_crop(image_bgr, _page_upscale(image_bgr.shape)),
            }
        ]
        recognized_text = regions[0]["text"]
    else:
        regions = _detect_and_ocr(image_bgr, model)
        full_image_text = _ocr_crop(image_bgr, _page_upscale(image_bgr.shape))
        layout_text = "\n".join(region["text"] for region in regions if region["text"])
        recognized_text = _merge_text(full_image_text, layout_text)

    return {
        "text": recognized_text,
        "regions": regions,
        "text_source": "layout_and_full_image" if model is not None else "full_image",
        "preprocessing": {
            "image_width": int(image_bgr.shape[1]),
            "image_height": int(image_bgr.shape[0]),
            "upscale": _page_upscale(image_bgr.shape),
            "white_margin_px": MARGIN,
            "psm": 6,
        },
    }

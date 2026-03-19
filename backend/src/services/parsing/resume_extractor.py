from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ResumeExtractionError(RuntimeError):
    pass


@dataclass
class ExtractedBlock:
    page: int
    bbox: dict[str, float]
    text: str


@dataclass
class ResumeExtractionResult:
    full_text: str
    blocks: list[ExtractedBlock]
    used_ocr_fallback: bool


def _run_ocr_fallback(_: Any) -> str:
    # OCR is intentionally a hook so a future engine can be plugged in.
    return ""


def extract_resume(payload: bytes) -> ResumeExtractionResult:
    if not payload:
        raise ResumeExtractionError("Empty file payload")

    try:
        import fitz
    except ImportError as exc:  # pragma: no cover
        raise ResumeExtractionError("PyMuPDF is required for resume extraction") from exc

    blocks: list[ExtractedBlock] = []
    aggregated_lines: list[str] = []
    used_ocr_fallback = False

    with fitz.open(stream=payload, filetype="pdf") as doc:
        for page_index in range(len(doc)):
            page = doc[page_index]
            page_blocks = page.get_text("blocks") or []
            for raw_block in page_blocks:
                x0, y0, x1, y1, text, *_rest = raw_block
                normalized = (text or "").strip()
                if not normalized:
                    continue
                blocks.append(
                    ExtractedBlock(
                        page=page_index + 1,
                        bbox={"x0": float(x0), "y0": float(y0), "x1": float(x1), "y1": float(y1)},
                        text=normalized,
                    )
                )
                aggregated_lines.append(normalized)

            if not page_blocks:
                fallback_text = _run_ocr_fallback(page)
                if fallback_text.strip():
                    used_ocr_fallback = True
                    blocks.append(
                        ExtractedBlock(
                            page=page_index + 1,
                            bbox={"x0": 0.0, "y0": 0.0, "x1": 0.0, "y1": 0.0},
                            text=fallback_text.strip(),
                        )
                    )
                    aggregated_lines.append(fallback_text.strip())

    full_text = "\n".join(aggregated_lines).strip()
    if not full_text:
        raise ResumeExtractionError("No readable text found in PDF")

    return ResumeExtractionResult(full_text=full_text, blocks=blocks, used_ocr_fallback=used_ocr_fallback)

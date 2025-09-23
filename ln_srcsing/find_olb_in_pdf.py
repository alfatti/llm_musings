import os
import io
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import fitz  # PyMuPDF
from PIL import Image
from rapidfuzz import fuzz, process

import google.generativeai as genai


# ----------------------------
# 1) CONFIG / VAULT INTEGRATION
# ----------------------------

def get_gemini_api_key_via_vault() -> str:
    """
    Replace this stub with the Vault retrieval you already use in your VLM pipelines.
    For example, your org's helper might look like:
        return read_secret_from_vault(path="kv/data/llm/google", key="GEMINI_API_KEY")
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set. Plug in your Vault retrieval in get_gemini_api_key_via_vault().")
    return api_key


MODEL_NAME = "gemini-2.0-flash"  # or "gemini-2.5-flash" if your org allows it
MAX_WORKERS = min(8, os.cpu_count() or 4)

# Variants to match the OLB label robustly
TARGET_LABELS = [
    "Outstanding Loan Balance",
    "Outstanding Loan Balance (OLB)",
    "OLB",
    "Outstanding Balance",
]


# ----------------------------
# 2) PDF -> PAGE IMAGES
# ----------------------------

def pdf_to_images(pdf_path: str, dpi: int = 220) -> List[Image.Image]:
    """
    Convert each page of a PDF into a Pillow Image.
    Returns a list of images in page order (0-based).
    """
    images = []
    doc = fitz.open(pdf_path)
    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        # zoom for higher OCR fidelity inside the VLM
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    doc.close()
    return images


# ----------------------------
# 3) GEMINI PROMPT ENGINEERING
# ----------------------------

SYSTEM_INSTRUCTIONS = """\
You are an expert financial analyst and document vision model specializing in spreadsheets printed to PDF with row/column headers visible.

Your task on EACH page image:
1) Find the row that contains a label matching the concept "Outstanding Loan Balance (OLB)". The label may appear as:
   - "Outstanding Loan Balance", "Outstanding Loan Balance (OLB)", "OLB", "Outstanding Balance"
   - Variants in capitalization, punctuation, or with nearby descriptors are acceptable.

2) Determine the VALUE for that label. Typical patterns:
   - Label in column A (or leftmost visible label column) and the numeric value in the same row, a few columns to the right.
   - The value should be the primary financial figure (e.g., 12,345,678.90). Prefer the most right-aligned numeric entry on that row if multiple numbers exist.

3) Output the Excel-like cell coordinate (e.g., "N23") for the VALUE cell (NOT the label cell).
   - Use the visible column headers at the top (A, B, C, ..., AA, AB, etc.) and visible row numbers at the left/right margins.
   - DO NOT guess beyond what is visible on the page. If column headers are partially visible, infer the column letter only if the sequence is unambiguous.
   - If unsure between two column letters, choose the more plausible and lower letter and reduce confidence.

4) When found, return a compact JSON **only** in this schema:
{
  "found": true,
  "page_index": <int, 0-based>,
  "value_text": "<string exactly as seen, including commas and decimals>",
  "cell_coordinate": "<Excel-like, e.g., N23>",
  "bbox": [x0, y0, x1, y1],  // approx pixel box of the VALUE cell in the given page image
  "label_variant": "<which label form you matched>",
  "confidence": <float between 0 and 1>,
  "evidence": "<brief justification>"
}

5) If not found on the page, return:
{ "found": false, "page_index": <int> }

Rules:
- Be precise and concise. Return JSON only. No markdown fences.
- Confidence reflects BOTH label match quality and coordinate certainty.
"""

def build_user_prompt(page_index: int) -> str:
    return f"""\
Document: Single loan sheet printed across multiple PDF pages in Z-order. This is page index {page_index} (0-based).
Find 'Outstanding Loan Balance (OLB)' value on THIS page and report JSON as specified.
"""


# ----------------------------
# 4) GEMINI CLIENT
# ----------------------------

def init_gemini_client():
    api_key = get_gemini_api_key_via_vault()
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        MODEL_NAME,
        system_instruction=SYSTEM_INSTRUCTIONS,
        generation_config={
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 40,
            "max_output_tokens": 1024,
            "response_mime_type": "application/json",
        },
        safety_settings=None,  # finance doc OCR is fine
    )
    return model


# ----------------------------
# 5) UTIL: IMAGE PACKAGING
# ----------------------------

def pil_image_to_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ----------------------------
# 6) CALL GEMINI ON ONE PAGE
# ----------------------------

@dataclass
class PageResult:
    page_index: int
    found: bool
    value_text: Optional[str] = None
    cell_coordinate: Optional[str] = None
    bbox: Optional[List[int]] = None
    label_variant: Optional[str] = None
    confidence: float = 0.0
    evidence: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def call_gemini_on_page(model, img: Image.Image, page_index: int) -> PageResult:
    """Run one VLM call on a single page image, parse robustly."""
    try:
        image_bytes = pil_image_to_bytes(img, fmt="PNG")
        prompt = build_user_prompt(page_index)

        # gemini expects a list of "parts": text + image
        response = model.generate_content(
            [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/png", "data": image_bytes}},
            ]
        )

        text = response.text or ""
        # Some org configs may still return code fences—strip them if present:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("Non-dict JSON returned")

        # Normalize
        found = bool(data.get("found", False))
        if not found:
            return PageResult(page_index=page_index, found=False, raw=data)

        # Ensure required fields
        value_text = data.get("value_text")
        cell_coordinate = data.get("cell_coordinate")
        confidence = float(data.get("confidence", 0.0) or 0.0)

        # Simple sanity check on label variant textual similarity
        label_variant = (data.get("label_variant") or "").strip()
        best_label, score, _ = process.extractOne(
            label_variant, TARGET_LABELS, scorer=fuzz.token_set_ratio
        )
        # If model hallucinated weird label, slightly dampen confidence
        if score < 60:
            confidence *= 0.9

        bbox = data.get("bbox")
        evidence = data.get("evidence")

        return PageResult(
            page_index=page_index,
            found=True,
            value_text=value_text,
            cell_coordinate=cell_coordinate,
            bbox=bbox if isinstance(bbox, list) else None,
            label_variant=label_variant,
            confidence=max(0.0, min(1.0, confidence)),
            evidence=evidence,
            raw=data,
        )
    except Exception as e:
        return PageResult(page_index=page_index, found=False, error=str(e))


# ----------------------------
# 7) ORCHESTRATION
# ----------------------------

def find_olb_in_pdf(pdf_path: str, max_workers: int = MAX_WORKERS) -> Dict[str, Any]:
    """
    Shred PDF -> images, run Gemini VLM per page in parallel, return best hit.
    """
    images = pdf_to_images(pdf_path)
    model = init_gemini_client()

    results: List[PageResult] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(call_gemini_on_page, model, img, idx): idx for idx, img in enumerate(images)}
        for fut in as_completed(futures):
            results.append(fut.result())

    # Pick best by confidence
    found_results = [r for r in results if r.found and r.value_text and r.cell_coordinate]
    if not found_results:
        return {
            "found": False,
            "message": "OLB not found on any page.",
            "per_page": [r.__dict__ for r in results],
        }

    found_results.sort(key=lambda r: r.confidence, reverse=True)
    best = found_results[0]

    return {
        "found": True,
        "page_index": best.page_index,
        "value_text": best.value_text,
        "cell_coordinate": best.cell_coordinate,
        "confidence": best.confidence,
        "label_variant": best.label_variant,
        "bbox": best.bbox,
        "evidence": best.evidence,
        "per_page": [r.__dict__ for r in results],  # keep all for audit
    }


# ----------------------------
# 8) CLI ENTRYPOINT (OPTIONAL)
# ----------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Locate Outstanding Loan Balance (OLB) in a printed-to-PDF loan sheet.")
    parser.add_argument("pdf_path", help="Path to the printed loan sheet PDF (single worksheet across pages).")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Max parallel VLM calls.")
    args = parser.parse_args()

    out = find_olb_in_pdf(args.pdf_path, max_workers=args.workers)
    print(json.dumps(out, indent=2))

# How it works (and why it’s robust)
# Coordinate extraction: The prompt forces Gemini to read the visible row/column headers from each page image—exactly what a human would do when a sheet is printed with headers. It then outputs the value cell’s coordinate (e.g., N23), not the label cell.
# Label variants: We accept several common forms of “Outstanding Loan Balance,” and we lightly re-score confidence if the label looks off.
# Parallel fan-out: Each page is processed concurrently. We return a per-page audit plus the best (highest confidence) hit.
# Z-order vs. page index: You don’t need to reconstruct the Z path here. We just report the page index (0-based) where Gemini found the cell. If you want to map that back to the printer’s “Page X of Y,” you can add an optional regex to read footer text.

# Notes:
# If your PDFs are huge or very high-DPI, consider reducing dpi to ~180 to save latency.
# If a page doesn’t show the column header row (e.g., “row/col headings repeat” was off), Gemini will lower confidence or return found=false for that page—exactly what we want.
# Want to double-check the coordinate? Add a post-step that crops the returned bbox and writes a debug image—handy in dev.

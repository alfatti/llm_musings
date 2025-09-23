import os
import io
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from pdf2image import convert_from_path
from PIL import Image
import google.generativeai as genai


# ----------------------------
# 1) CONFIG / VAULT INTEGRATION
# ----------------------------

def get_gemini_api_key_via_vault() -> str:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set. Plug in your Vault retrieval in get_gemini_api_key_via_vault().")
    return api_key

MODEL_NAME = "gemini-2.0-flash"  # or "gemini-2.5-flash" if available
MAX_WORKERS = min(8, os.cpu_count() or 4)

# The ONLY acceptable normalized labels. The model must choose exactly one or declare not found.
ALLOWED_LABELS = [
    "Outstanding Loan Balance",
    "Outstanding Loan Balance (OLB)",
    "OLB",
    "Outstanding Balance",
]


# ----------------------------
# 2) PDF -> PAGE IMAGES (pdf2image)
# ----------------------------

def pdf_to_images(pdf_path: str, dpi: int = 220) -> List[Image.Image]:
    return convert_from_path(pdf_path, dpi=dpi)


# ----------------------------
# 3) GEMINI PROMPT ENGINEERING
# ----------------------------

SYSTEM_INSTRUCTIONS = f"""\
You are an expert financial analyst and document vision model for spreadsheets printed to PDF with row/column headers visible.

Your job on EACH single page image:
1) Identify a row whose label refers to the concept "Outstanding Loan Balance (OLB)". The label may appear with typographical variation.
2) Read the VALUE cell for that row (the numeric figure) and return its Excel-like cell coordinate (e.g., "N23"). Use the visible column letters and row numbers on the page. Do not guess beyond what is visible; if the column sequence is partially visible, infer only if unambiguous.
3) You MUST normalize the detected label to EXACTLY ONE of the following allowed options, or declare not found:
   - {ALLOWED_LABELS[0]}
   - {ALLOWED_LABELS[1]}
   - {ALLOWED_LABELS[2]}
   - {ALLOWED_LABELS[3]}

Normalization rules:
- If the on-page label ("label_variant_seen") is a near-variant (e.g., punctuation/casing differences, abbreviations like "Outstanding Ln Bal."), you MUST map it to the closest correct item from the allowed list ("label_normalized").
- If you cannot confidently map to one of the allowed options, return found=false for this page.

Output STRICT JSON only. No markdown fences. Use this schema:
{{
  "found": true|false,
  "page_index": <int, 0-based>,
  "value_text": "<string exactly as seen, including commas/decimals, required if found>",
  "cell_coordinate": "<Excel-like, e.g., N23, required if found>",
  "bbox": [x0, y0, x1, y1],  // approximate pixel bounding box of the VALUE cell on this page (ints)
  "label_variant_seen": "<the label text as it appears on the page>",
  "label_normalized": "<one of the ALLOWED_LABELS exactly>",
  "confidence": <float 0..1>,  // reflect BOTH label normalization certainty and coordinate certainty
  "evidence": "<brief justification>"
}}

Additional rules:
- The VALUE cell is the numeric figure for Outstanding Loan Balance. Prefer the right-aligned principal amount if multiple numbers are on the row.
- If not found on this page, return: {{ "found": false, "page_index": <int> }}.
- Be precise and concise. Return JSON only.
"""

def build_user_prompt(page_index: int) -> str:
    return f"""\
Document: Single loan sheet printed across multiple PDF pages in Z-order. This is page index {page_index} (0-based).
Task: Find 'Outstanding Loan Balance (OLB)' value on THIS page and return JSON exactly per schema.
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
        safety_settings=None,
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
    label_variant_seen: Optional[str] = None
    label_normalized: Optional[str] = None
    confidence: float = 0.0
    evidence: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def call_gemini_on_page(model, img: Image.Image, page_index: int) -> PageResult:
    try:
        image_bytes = pil_image_to_bytes(img, fmt="PNG")
        prompt = build_user_prompt(page_index)

        response = model.generate_content(
            [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/png", "data": image_bytes}},
            ]
        )

        text = (response.text or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("Non-dict JSON returned")

        if not data.get("found", False):
            return PageResult(page_index=page_index, found=False, raw=data)

        return PageResult(
            page_index=page_index,
            found=True,
            value_text=data.get("value_text"),
            cell_coordinate=data.get("cell_coordinate"),
            bbox=data.get("bbox") if isinstance(data.get("bbox"), list) else None,
            label_variant_seen=data.get("label_variant_seen"),
            label_normalized=data.get("label_normalized"),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            evidence=data.get("evidence"),
            raw=data,
        )
    except Exception as e:
        return PageResult(page_index=page_index, found=False, error=str(e))


# ----------------------------
# 7) ORCHESTRATION
# ----------------------------

def find_olb_in_pdf(pdf_path: str, max_workers: int = MAX_WORKERS) -> Dict[str, Any]:
    images = pdf_to_images(pdf_path)
    model = init_gemini_client()

    results: List[PageResult] = []
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(call_gemini_on_page, model, img, idx): idx for idx, img in enumerate(images)}
        for fut in as_completed(futures):
            results.append(fut.result())

    found_results = [
        r for r in results
        if r.found and r.value_text and r.cell_coordinate and r.label_normalized in ALLOWED_LABELS
    ]
    if not found_results:
        return {
            "found": False,
            "message": "OLB not found on any page.",
            "per_page": [r.__dict__ for r in results],
        }

    # Pick the highest confidence.
    found_results.sort(key=lambda r: r.confidence, reverse=True)
    best = found_results[0]
    return {
        "found": True,
        "page_index": best.page_index,
        "value_text": best.value_text,
        "cell_coordinate": best.cell_coordinate,
        "label_variant_seen": best.label_variant_seen,
        "label_normalized": best.label_normalized,
        "confidence": best.confidence,
        "bbox": best.bbox,
        "evidence": best.evidence,
        "per_page": [r.__dict__ for r in results],
    }


# ----------------------------
# 8) CLI ENTRYPOINT
# ----------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Locate Outstanding Loan Balance (OLB) in a printed-to-PDF loan sheet.")
    parser.add_argument("pdf_path", help="Path to the printed loan sheet PDF (single worksheet across pages).")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Max parallel VLM calls.")
    args = parser.parse_args()

    out = find_olb_in_pdf(args.pdf_path, max_workers=args.workers)
    print(json.dumps(out, indent=2))

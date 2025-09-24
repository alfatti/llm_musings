import os
import io
import json
import re
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import requests
from pdf2image import convert_from_path
from PIL import Image


# ----------------------------
# 1) CONFIG
# ----------------------------

# You said you already have these:
API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()  # or set it however you like
MODEL_ENDPOINT = os.environ.get("GEMINI_MODEL_ENDPOINT", "").strip()
# Example:
# MODEL_ENDPOINT = "https://us-central-aiplatform.googleapis.com/v1/projects/<proj>/locations/us-central1/publishers/google/models/gemini-2.5-flash:generateContent"

if not API_KEY or not MODEL_ENDPOINT:
    raise RuntimeError("Set GEMINI_API_KEY and GEMINI_MODEL_ENDPOINT (full :generateContent URL).")

MAX_WORKERS = min(8, os.cpu_count() or 4)

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
# 3) PROMPT
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
  "bbox": [x0, y0, x1, y1],
  "label_variant_seen": "<the label text as it appears on the page>",
  "label_normalized": "<one of the ALLOWED_LABELS exactly>",
  "confidence": <float 0..1>,
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
# 4) REST CLIENT (Vertex AI)
# ----------------------------

HEADERS = {
    # Either use API key as header...
    "x-goog-api-key": API_KEY,
    # ...or append ?key=API_KEY to the URL instead (shown below in _post()).
    "Content-Type": "application/json; charset=utf-8",
}

GENERATION_CONFIG = {
    "temperature": 0.0,
    "topP": 1.0,
    "topK": 40,
    "maxOutputTokens": 1024,
    "responseMimeType": "application/json",
}

def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")

def _post(payload: Dict[str, Any]) -> Dict[str, Any]:
    # You can pass the API key either via header (above) or query param.
    url = f"{MODEL_ENDPOINT}?key={API_KEY}"
    resp = requests.post(url, headers=HEADERS, data=json.dumps(payload), timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"Vertex AI error {resp.status_code}: {resp.text}")
    return resp.json()

def call_gemini_rest(prompt_text: str, image_bytes: bytes) -> str:
    """
    Calls Vertex AI's :generateContent via REST with a single image and prompt.
    Returns the model text (expected to be JSON per our instructions).
    """
    payload = {
        "systemInstruction": {
            "role": "system",
            "parts": [{"text": SYSTEM_INSTRUCTIONS}],
        },
        "generationConfig": GENERATION_CONFIG,
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt_text},
                    {
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": _b64(image_bytes),
                        }
                    },
                ],
            }
        ],
    }

    data = _post(payload)
    # Response shape: candidates[0].content.parts[0].text
    try:
        candidates = data.get("candidates", [])
        parts = (
            candidates[0]
            .get("content", {})
            .get("parts", [])
        )
        text = parts[0].get("text", "") if parts else ""
        return text
    except Exception:
        # If the model used "promptFeedback" or other structure, surface it:
        return json.dumps(data)


# ----------------------------
# 5) UTIL
# ----------------------------

def pil_image_to_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ----------------------------
# 6) PER-PAGE CALL/RESULT
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

def call_llm_on_page(img: Image.Image, page_index: int) -> PageResult:
    try:
        image_bytes = pil_image_to_bytes(img, fmt="PNG")
        prompt = build_user_prompt(page_index)

        text = (call_gemini_rest(prompt, image_bytes) or "").strip()
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

    results: List[PageResult] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(call_llm_on_page, img, idx): idx for idx, img in enumerate(images)}
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
# 8) CLI
# ----------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Locate Outstanding Loan Balance (OLB) in a printed-to-PDF loan sheet.")
    parser.add_argument("pdf_path", help="Path to the printed loan sheet PDF (single worksheet across pages).")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Max parallel LLM calls.")
    args = parser.parse_args()

    out = find_olb_in_pdf(args.pdf_path, max_workers=args.workers)
    print(json.dumps(out, indent=2))

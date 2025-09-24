#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import re
import json
import base64
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from pdf2image import convert_from_path
from PIL import Image


# =========================
# 0) CONFIG / CONSTANTS
# =========================

# Required env vars:
#   GEMINI_API_KEY  -> Bearer token for Vertex/Google endpoint
#   GEMINI_ENDPOINT -> Full :generateContent URL
# Example:
#   export GEMINI_ENDPOINT="https://us-central-aiplatform.googleapis.com/v1/projects/<proj>/locations/us-central1/publishers/google/models/gemini-2.5-flash:generateContent"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_ENDPOINT = os.environ.get("GEMINI_ENDPOINT", "").strip()
if not GEMINI_API_KEY or not GEMINI_ENDPOINT:
    raise RuntimeError("Set GEMINI_API_KEY and GEMINI_ENDPOINT environment variables.")

MAX_WORKERS = min(8, os.cpu_count() or 4)

# Vertex generation config (tune if needed)
GENERATION_CONFIG = {
    "temperature": 0.0,
    "topP": 1.0,
    "topK": 40,
    "maxOutputTokens": 1024,
    "responseMimeType": "application/json",
}

# Variable → allowed label variations
VARIABLE_ALIASES: Dict[str, List[str]] = {
    "Outstanding Loan Amount": [
        "Outstanding Loan Amount",
        "Outstanding Loan Balance",
        "Outstanding Balance",
        "OLB",
        "Loan Outstanding",
        "Outstanding Principal Balance",
        "Curr. Outstanding",
    ],
    "Cash-Interest": [
        "Cash-Interest",
        "Cash Interest",
        "Interest (Cash)",
        "Interest Paid (Cash)",
        "Interest – Cash",
    ],
    "Cash-Principal": [
        "Cash-Principal",
        "Cash Principal",
        "Principal (Cash)",
        "Principal Paid (Cash)",
        "Principal – Cash",
    ],
    "Cash-Other": [
        "Cash-Other",
        "Cash Other",
        "Other (Cash)",
        "Other Cash",
        "Misc Cash",
    ],
    "Total Loans PIK": [
        "Total Loans PIK",
        "PIK Total",
        "Total PIK",
        "Loans PIK Total",
        "PIK (Loans) Total",
    ],
}

# System message (generic, variable awareness pushed to user prompt)
SYSTEM_INSTRUCTIONS = """\
You are an expert financial analyst and document vision model for spreadsheets printed to PDF with row/column headers visible.

On EACH single page image:
1) Locate the row corresponding to the requested VARIABLE (given in the user prompt). The label may appear with typographical variation.
2) Read the VALUE cell for that row (the numeric figure) and return its Excel-like cell coordinate (e.g., "N23"). Use only headers visible on the page; if the sequence is partially visible, infer only if unambiguous.
3) Normalize the detected on-page label to EXACTLY ONE of the allowed options provided in the user prompt; otherwise return found=false.

Output STRICT JSON only. No markdown fences. Schema:
{
  "found": true|false,
  "page_index": <int, 0-based>,
  "variable": "<the requested variable name>",
  "value_text": "<string exactly as seen, incl. commas/decimals, required if found>",
  "cell_coordinate": "<Excel-like, e.g., N23, required if found>",
  "bbox": [x0, y0, x1, y1],
  "label_variant_seen": "<label as it appears on the page>",
  "label_normalized": "<one of the allowed options exactly>",
  "confidence": <float 0..1>,
  "evidence": "<brief justification>"
}

Rules:
- Prefer the right-aligned principal numeric on the row if multiple numbers exist.
- If not found on this page, return: { "found": false, "page_index": <int> }.
- Return JSON only.
"""


# =========================
# 1) PDF → IMAGES
# =========================

def pdf_to_images(pdf_path: str, dpi: int = 220) -> List[Image.Image]:
    """
    Convert each page of a PDF into a Pillow Image using pdf2image.
    Requires Poppler installed on the system.
    """
    return convert_from_path(pdf_path, dpi=dpi)


# =========================
# 2) PROMPT MAKER
# =========================

def make_prompt(page_index: int, variable: str, allowed_labels: List[str]) -> str:
    allowed_bullets = "\n   - " + "\n   - ".join(allowed_labels)
    return f"""\
Document: Single loan sheet printed across multiple PDF pages (Z-order). This is page index {page_index} (0-based).

VARIABLE: "{variable}"

You MUST normalize the on-page label to EXACTLY one of the following allowed options, or declare not found on this page:
{allowed_bullets}

Task: Find the VALUE for VARIABLE on THIS page and return JSON exactly per the schema in the system message.
"""


# =========================
# 3) LOW-LEVEL REST CALL
# =========================

def image_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def extract_from_image(img: Image.Image, user_prompt: str) -> str:
    """
    Build request body, POST via requests to Vertex/Google :generateContent endpoint,
    return the model's text (expected to be JSON per our prompt).
    """
    image_b64 = image_to_base64(img, fmt="PNG")

    request_body = {
        "systemInstruction": {
            "role": "system",
            "parts": [{"text": SYSTEM_INSTRUCTIONS}],
        },
        "generationConfig": GENERATION_CONFIG,
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": user_prompt},
                    {"inlineData": {"mimeType": "image/png", "data": image_b64}},
                ],
            }
        ],
    }

    headers = {
        "Authorization": f"Bearer {GEMINI_API_KEY}",
        "Content-Type": "application/json; charset=utf-8",
    }

    resp = requests.post(GEMINI_ENDPOINT, headers=headers, json=request_body, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    # Typical Vertex response shape: candidates[0].content.parts[0].text
    candidates = data.get("candidates", [])
    parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
    text = parts[0].get("text", "") if parts else ""
    return text or json.dumps(data)


# =========================
# 4) PAGE CALL + RESULT MODEL
# =========================

@dataclass
class PageResult:
    page_index: int
    found: bool
    variable: Optional[str] = None
    value_text: Optional[str] = None
    cell_coordinate: Optional[str] = None
    bbox: Optional[List[int]] = None
    label_variant_seen: Optional[str] = None
    label_normalized: Optional[str] = None
    confidence: float = 0.0
    evidence: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

def call_llm_on_page(img: Image.Image, page_index: int, variable: str, allowed_labels: List[str]) -> PageResult:
    try:
        prompt = make_prompt(page_index, variable, allowed_labels)
        text = (extract_from_image(img, prompt) or "").strip()

        # Strip accidental code fences if any
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("Non-dict JSON returned from model.")

        if not data.get("found", False):
            return PageResult(page_index=page_index, found=False, variable=variable, raw=data)

        return PageResult(
            page_index=page_index,
            found=True,
            variable=data.get("variable") or variable,
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
        return PageResult(page_index=page_index, found=False, variable=variable, error=str(e))


# =========================
# 5) ORCHESTRATION
# =========================

def find_variable_in_pdf(pdf_path: str, variable: str, max_workers: int = MAX_WORKERS) -> Dict[str, Any]:
    """
    Shreds a printed-to-PDF worksheet into images, queries Gemini per page,
    and returns the best match (by confidence) for the requested variable.
    """
    allowed = VARIABLE_ALIASES.get(variable)
    if not allowed:
        raise ValueError(f'No alias list registered for variable "{variable}". Add it to VARIABLE_ALIASES.')

    images = pdf_to_images(pdf_path)

    results: List[PageResult] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(call_llm_on_page, img, idx, variable, allowed): idx
            for idx, img in enumerate(images)
        }
        for fut in as_completed(futures):
            results.append(fut.result())

    # Keep only good hits normalized to one of the allowed labels
    found_results = [
        r for r in results
        if r.found and r.value_text and r.cell_coordinate and (r.label_normalized in allowed)
    ]
    if not found_results:
        return {
            "found": False,
            "variable": variable,
            "message": f'"{variable}" not found on any page.',
            "per_page": [r.__dict__ for r in results],
        }

    # Pick highest confidence
    found_results.sort(key=lambda r: r.confidence, reverse=True)
    best = found_results[0]
    return {
        "found": True,
        "variable": variable,
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


# =========================
# 6) CLI
# =========================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract a variable from a printed-to-PDF loan sheet using Gemini VLM.")
    parser.add_argument("pdf_path", help="Path to the PDF (single worksheet printed across pages).")
    parser.add_argument("--variable", required=True, choices=list(VARIABLE_ALIASES.keys()),
                        help="Variable to extract (must exist in VARIABLE_ALIASES).")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Max parallel LLM calls.")
    args = parser.parse_args()

    out = find_variable_in_pdf(args.pdf_path, variable=args.variable, max_workers=args.workers)
    print(json.dumps(out, indent=2))

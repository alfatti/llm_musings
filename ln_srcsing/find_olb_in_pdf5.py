#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import re
import json
import base64
import argparse
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from pdf2image import convert_from_path
from PIL import Image


# ============================================================
# 0) CONFIG
# ============================================================

# Auth + endpoint (Vertex "generateContent")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_ENDPOINT = os.environ.get("GEMINI_ENDPOINT", "").strip()
# Example:
#   export GEMINI_ENDPOINT="https://us-central-aiplatform.googleapis.com/v1/projects/<proj>/locations/us-central1/publishers/google/models/gemini-2.5-flash:generateContent"

if not GEMINI_API_KEY or not GEMINI_ENDPOINT:
    raise RuntimeError("Set GEMINI_API_KEY and GEMINI_ENDPOINT (full :generateContent URL).")

MODEL_MAX_WORKERS = max(1, min(8, os.cpu_count() or 4))

# Fixed authoritative labels for audit (this is NOT the variability list — see HINTS below)
ALLOWED_LABELS = [
    "Outstanding Loan Balance",
    "Outstanding Loan Balance (OLB)",
    "OLB",
    "Outstanding Balance",
]

# Generation config
GENERATION_CONFIG = {
    "temperature": 0.0,
    "topP": 1.0,
    "topK": 40,
    "maxOutputTokens": 1024,
    "responseMimeType": "application/json",
}


# ============================================================
# 1) DEFAULT HINTS (you can override via --hints_file)
# ============================================================

DEFAULT_HINTS = {
    "Outstanding Loan Balance (OLB)": {
        # Boxed regions / panels
        "sections": [
            "Loan Summary",
            "Balance Overview",
            "Debt Summary",
            "Loan Information",
            "Collateral / Loan Summary",
        ],
        # Column / subtable names (sub-sections)
        "sub_sections": [
            "Principal Balance",
            "Loan Balance",
            "Balances",
            "Outstanding",
            "Totals",
        ],
        # Row line-items used to locate the actual value cell
        "line_items": [
            "Outstanding Loan Balance",
            "Total Outstanding",
            "Aggregate OLB",
            "OLB",
            "Total Balance",
            "Outstanding Balance",
        ],
    }
}


# ============================================================
# 2) PDF -> IMAGES
# ============================================================

def pdf_to_images(pdf_path: str, dpi: int = 220) -> List[Image.Image]:
    """
    Convert each PDF page to a PIL Image.
    NOTE: Requires Poppler installed (pdf2image dependency).
    """
    return convert_from_path(pdf_path, dpi=dpi)


# ============================================================
# 3) PROMPT CONSTRUCTION (schema-aware)
# ============================================================

def build_system_instructions() -> str:
    """
    System message describes the complex, boxy Excel-to-PDF layout and the audit JSON we want.
    NOTE: ALLOWED_LABELS are authoritative for label_normalized only (audit field).
    The schema hints are variability cues, NOT normalization guidance.
    """
    allowed_list = "\n   - ".join(ALLOWED_LABELS)
    return f"""\
You are an expert financial analyst and document vision model for spreadsheets printed to PDF with row/column headers visible.

Context:
- The PDF is produced by printing a single Excel sheet across multiple pages (Z-shaped coverage). Each page may show different regions with repeated row/column headers.
- The layout is often NOT a single flat table; there may be multiple boxed regions ("sections"). Within a section there may be sub-sections (columns or sub-tables). The target value is usually found on a specific line-item (row), frequently a sum/total near the bottom (e.g., 'Total', 'Aggregate').

Your job on EACH page image:
1) Use the provided schema hints to guide search across sections → sub-sections (columns) → line-items (rows).
   - Hints are variability cues ONLY. They indicate how names may differ across workbooks.
   - Do NOT rename or normalize from hints; use them to navigate to the right area and row.

2) Identify the region most relevant to the target variable; find the VALUE cell on the same row as the target line-item.
   - The VALUE is the main numeric figure on that row. If multiple numbers exist, prefer the right-aligned principal/total amount.

3) Return the Excel-like cell coordinate (e.g., "N23") of the VALUE cell (NOT the label cell).
   - Use visible column letters and row numbers on the page. Do not guess beyond what is visible; infer sequences only if unambiguous.

4) For audit purposes, set "label_normalized" to EXACTLY one item from this fixed list:
   - {allowed_list}

Output STRICT JSON only. No markdown fences. Use this schema:
{{
  "found": true|false,
  "page_index": <int, 0-based>,
  "value_text": "<string exactly as seen, including commas/decimals, required if found>",
  "cell_coordinate": "<Excel-like, e.g., N23, required if found>",
  "bbox": [x0, y0, x1, y1],  // approximate pixel bounds of the VALUE cell on this page
  "label_variant_seen": "<the label text as it appears on the page>",
  "label_normalized": "<one of the fixed labels above>",
  "confidence": <float 0..1>,  // reflect BOTH label certainty and cell coordinate certainty
  "evidence": "<brief justification>"
}}

Rules:
- If not found on this page, return: {{ "found": false, "page_index": <int> }}.
- Be precise and concise. Return JSON only.
"""


def build_user_prompt(page_index: int, variable_key: str, hints: Dict[str, Any]) -> str:
    """
    Build a page-specific user prompt that injects the variable name and its schema hints.
    The hints are indicators of naming variability (NOT normalization instructions).
    """
    var_hints = hints.get(variable_key, {"sections": [], "sub_sections": [], "line_items": []})
    hints_json = json.dumps(
        {
            "variable": variable_key,
            "hints": {
                "sections": var_hints.get("sections", []),
                "sub_sections": var_hints.get("sub_sections", []),
                "line_items": var_hints.get("line_items", []),
            }
        },
        ensure_ascii=False
    )
    return (
        f"Document: Single loan sheet printed across multiple PDF pages in Z-order. "
        f"This is page index {page_index} (0-based).\n"
        f"Target variable: {variable_key}\n"
        f"Schema hints (variability cues only): {hints_json}\n"
        f"Task: Using these hints to navigate among sections, sub-sections (columns), and line-items (rows), "
        f"find the VALUE cell for the target variable on THIS page and return JSON exactly per schema."
    )


# ============================================================
# 4) LOW-LEVEL REST CALL
# ============================================================

def image_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def extract_from_image(img: Image.Image, user_prompt: str, system_instructions: str) -> str:
    """
    Build request body, POST to Vertex, return model's raw text (expected JSON per instructions).
    """
    image_b64 = image_to_base64(img, fmt="PNG")

    request_body = {
        "systemInstruction": {
            "role": "system",
            "parts": [{"text": system_instructions}],
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

    # Typical Vertex response shape:
    #   data["candidates"][0]["content"]["parts"][0]["text"]
    candidates = data.get("candidates", [])
    parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
    text = parts[0].get("text", "") if parts else ""
    return text or json.dumps(data)


# ============================================================
# 5) PAGE CALL + RESULT MODEL
# ============================================================

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


def call_llm_on_page(img: Image.Image, page_index: int, variable_key: str, hints: Dict[str, Any], system_instructions: str) -> PageResult:
    try:
        prompt = build_user_prompt(page_index, variable_key, hints)
        text = (extract_from_image(img, prompt, system_instructions) or "").strip()

        # strip accidental code fences if any
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


# ============================================================
# 6) ORCHESTRATION
# ============================================================

def find_variable_in_pdf(
    pdf_path: str,
    variable_key: str = "Outstanding Loan Balance (OLB)",
    hints: Optional[Dict[str, Any]] = None,
    max_workers: int = MODEL_MAX_WORKERS,
) -> Dict[str, Any]:
    """
    Shred PDF -> images, run Gemini VLM per page in parallel, return best hit and audit.
    """
    images = pdf_to_images(pdf_path)
    hints = hints or DEFAULT_HINTS
    system_instructions = build_system_instructions()

    results: List[PageResult] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(call_llm_on_page, img, idx, variable_key, hints, system_instructions): idx
            for idx, img in enumerate(images)
        }
        for fut in as_completed(futures):
            results.append(fut.result())

    # Accept only results that also set label_normalized to one of our fixed audit labels.
    found_results = [
        r for r in results
        if r.found and r.value_text and r.cell_coordinate and (r.label_normalized in ALLOWED_LABELS)
    ]
    if not found_results:
        return {
            "found": False,
            "message": f"{variable_key} not found on any page.",
            "per_page": [r.__dict__ for r in results],
        }

    # Choose best by confidence
    found_results.sort(key=lambda r: r.confidence, reverse=True)
    best = found_results[0]

    return {
        "found": True,
        "variable_key": variable_key,
        "page_index": best.page_index,
        "value_text": best.value_text,
        "cell_coordinate": best.cell_coordinate,
        "label_variant_seen": best.label_variant_seen,
        "label_normalized": best.label_normalized,
        "confidence": best.confidence,
        "bbox": best.bbox,
        "evidence": best.evidence,
        "per_page": [r.__dict__ for r in results],  # full audit trail
    }


# ============================================================
# 7) CLI
# ============================================================

def main():
    p = argparse.ArgumentParser(description="Locate a variable (e.g., OLB) in a printed-to-PDF loan sheet using Gemini VLM (REST).")
    p.add_argument("pdf_path", help="Path to the printed loan sheet PDF (single worksheet across pages).")
    p.add_argument("--variable_key", default="Outstanding Loan Balance (OLB)", help="Name of the target variable (must exist in hints).")
    p.add_argument("--hints_file", default="", help="Path to a JSON file with hints dict to override defaults.")
    p.add_argument("--workers", type=int, default=MODEL_MAX_WORKERS, help="Max parallel LLM calls (default: auto).")
    args = p.parse_args()

    hints = DEFAULT_HINTS
    if args.hints_file:
        with open(args.hints_file, "r", encoding="utf-8") as f:
            hints = json.load(f)

    out = find_variable_in_pdf(
        pdf_path=args.pdf_path,
        variable_key=args.variable_key,
        hints=hints,
        max_workers=args.workers,
    )
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

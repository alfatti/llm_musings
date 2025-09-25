#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schema-aware PDF → JSON extractor using a VLM (Gemini via REST).
- Converts PDF pages to images
- Builds hierarchy-aware prompts (sections → sub-sections/columns → line-items)
- Parallelizes per-page VLM calls with ThreadPoolExecutor
- Chooses the best match across pages and returns clean JSON

ENV:
  GEMINI_API_KEY   : required (public Generative Language API)
  GEMINI_ENDPOINT  : optional override. Default uses v1beta generateContent:
                     https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
USAGE (single variable):
  python schema_aware_vlm_extractor.py ./sample.pdf --variable "Outstanding Loan Balance"

USAGE (multiple variables):
  python schema_aware_vlm_extractor.py ./sample.pdf --variable "Outstanding Loan Balance" --variable "Cash-Interest"

Notes:
- Requires: pdf2image (and poppler on your system), Pillow, requests
- Default model: gemini-1.5-flash
"""
import os
import io
import re
import json
import base64
import logging
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from pdf2image import convert_from_path
from PIL import Image

# ---------------------------
# Defaults & Config
# ---------------------------

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
DEFAULT_ENDPOINT_TMPL = os.environ.get(
    "GEMINI_ENDPOINT",
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
)
MAX_WORKERS = int(os.environ.get("VLM_MAX_WORKERS", "6"))
TIMEOUT_SEC = int(os.environ.get("VLM_TIMEOUT_SEC", "60"))

# Hierarchical variants dictionary: variable → {sections[], subsections[], line_items[], total_words[], avoid_words[], units[], notes}
VARIABLES_CFG: Dict[str, Dict[str, List[str]]] = {
    "Outstanding Loan Balance": {
        "sections": [
            "Loan Summary", "Collateral Summary", "Credit Summary", "Facility Summary",
            "Capitalization", "Debt Summary"
        ],
        "subsections": [
            "Outstanding", "Balance", "Principal Balance", "Ending Balance", "Ending Principal"
        ],
        "line_items": [
            "Outstanding Loan Balance", "OLB", "Total Outstanding", "Ending Principal Balance",
            "Outstanding Balance", "Loan Balance", "Ending Balance"
        ],
        "total_words": ["total", "aggregate", "sum", "subtotal", "ending total"],
        "avoid_words": ["undrawn", "unused", "commitment", "revolver availability"],
        "units": ["USD", "$", "millions", "mm", "thousands", "k"],
        "notes": ["Usually a period-end total; prefer totals over per-loan rows."],
    },
    "Cash-Interest": {
        "sections": ["Cash Flows", "Statement of Cash Flows", "Debt Service", "Interest"],
        "subsections": ["Cash Interest", "Interest Paid", "Interest Expense (Cash)"],
        "line_items": [
            "Cash Interest", "Interest Paid", "Interest Expense - Cash", "Interest (Cash)"
        ],
        "total_words": ["total", "aggregate", "sum"],
        "avoid_words": ["accrued", "PIK", "non-cash", "capitalized"],
        "units": ["USD", "$", "mm", "k"],
        "notes": ["Exclude accrued/non-cash, PIK."],
    },
    "Cash-Principal": {
        "sections": ["Debt Service", "Cash Flows", "Amortization"],
        "subsections": ["Principal", "Amortization", "Cash Principal"],
        "line_items": ["Principal Paid", "Cash Principal", "Debt Principal Payment", "Amortization"],
        "total_words": ["total", "aggregate", "sum"],
        "avoid_words": ["PIK", "non-cash"],
        "units": ["USD", "$", "mm", "k"],
        "notes": ["Exclude PIK/non-cash."],
    },
    "Cash-Other": {
        "sections": ["Debt Service", "Cash Flows"],
        "subsections": ["Other", "Fees", "Charges"],
        "line_items": ["Other Cash", "Fees Paid", "Other Debt Service"],
        "total_words": ["total", "aggregate", "sum"],
        "avoid_words": ["PIK", "non-cash"],
        "units": ["USD", "$", "mm", "k"],
        "notes": ["Catch-all cash items other than interest/principal."],
    },
    "Loan PIK": {
        "sections": ["Debt Service", "Interest", "Notes to Financials"],
        "subsections": ["PIK", "Non-cash Interest", "Capitalized Interest"],
        "line_items": ["PIK Interest", "Payment-in-kind", "Non-cash Interest", "Capitalized Interest"],
        "total_words": ["total", "aggregate", "sum"],
        "avoid_words": ["cash"],
        "units": ["USD", "$", "mm", "k"],
        "notes": ["Explicitly non-cash; exclude cash interest."],
    },
}

# ---------------------------
# Prompt Builder (schema-aware)
# ---------------------------

def build_schema_aware_prompt(
    *, pdf_path: str, variable: str, cfg: Dict[str, Any],
    page_range: Optional[List[int]] = None,
    period_hint: Optional[str] = None,
    prefer_current_period: bool = True,
    return_top_k: int = 1,
    require_coordinates: bool = True,
) -> Dict[str, str]:
    """
    Build system + user + context_json for a schema-aware extraction.
    Returns a dict with keys: system, user, context_json
    """
    if variable not in cfg:
        raise KeyError(f"Unknown variable '{variable}'. Add it to VARIABLES_CFG.")

    vc = cfg[variable]
    # ensure all fields exist
    sections = vc.get("sections", [])
    subsections = vc.get("subsections", [])
    line_items = vc.get("line_items", [])
    total_words = vc.get("total_words", [])
    avoid_words = vc.get("avoid_words", [])
    units = vc.get("units", [])
    notes = vc.get("notes", [])

    page_hint = f"Pages to prefer: {page_range}." if page_range else "Pages: search all."
    period_rule = (
        "If multiple periods are present, prefer the current/latest period totals."
        if prefer_current_period else
        "Do not prefer latest period; choose the best semantic match."
    )
    period_hint_text = f"Period hint: {period_hint}." if period_hint else "No explicit period hint."

    # JSON schema description (for the model; not a strict machine schema)
    schema_desc = {
        "variable": "string  # echo requested variable",
        "found": "boolean",
        "value": "number|null  # normalized numeric (strip thousands separators)",
        "unit": "string|null    # '$', 'USD', 'mm', 'k'",
        "scale_applied": "string|null  # 'millions' if doc says 'in millions' etc.",
        "sign": "string|null     # '+', '-' or null",
        "page_number": "integer|null  # 1-based page index if known",
        "bbox": "[x0,y0,x1,y1]|null   # bounding box for chosen value if available",
        "section_hit": "string|null",
        "subsection_hit": "string|null",
        "line_item_hit": "string|null",
        "evidence_text": "string|null  # short local quote near the value",
        "confidence": "number  # 0..1",
        "disambiguation": [
            {
                "value": "number",
                "page_number": "integer",
                "section_hit": "string|null",
                "subsection_hit": "string|null",
                "line_item_hit": "string|null",
                "evidence_text": "string"
            }
        ][:return_top_k],
        "not_found_reason": "string|null",
        "warnings": "string[]"
    }

    system_instructions = f"""
You are a meticulous financial document analyst. The input is a PDF print of a fixed-format spreadsheet.
The sheet may have 'sections' (sub-boxes), 'sub-sections' (columns/sub-headers), and 'line-items' (row labels).
Values we need are often totals at the bottom/right of a table.

Your job: extract exactly ONE best value for the requested variable using a hierarchical search:
  1) Find candidate SECTION headers first (if any).
  2) Within a chosen section, locate matching SUB-SECTION (column/sub-header) if present.
  3) Within that column, look for a LINE-ITEM row name that matches the variants (or synonyms).
  4) Prefer sum/total/aggregate rows when applicable.
  5) Normalize the numeric value and report its units/scale (e.g., 'in millions').

{page_hint}
{period_hint_text}
{period_rule}

Handle messy layouts: tables may be skewed, headers may be multi-line, totals may appear as 'Total', 'Aggregate', 'Sum'.
Use 'avoid_words' to filter out the wrong family (e.g., exclude 'PIK' for cash interest).

Output rules (CRITICAL):
  - Respond with STRICT JSON ONLY. No backticks, no markdown, no commentary.
  - Include 'evidence_text' (short span around the value/label) but do NOT include your reasoning.
  - If coordinates are available, include 'bbox' and 'page_number'. Otherwise set them to null.
  - If multiple candidates tie, choose the best one and list alternates in 'disambiguation'.

JSON fields (types/notes):
{json.dumps(schema_desc, indent=2)}
""".strip()

    user_message = f"""
Extract the value for the variable below from the attached PDF image.

VARIABLE: {variable}

SECTION variants (if present): {sections}
SUB-SECTION (column) variants: {subsections}
LINE-ITEM (row) variants: {line_items}
PREFER sum/total words: {total_words}
AVOID when text contains: {avoid_words}
EXPECTED units (if stated): {units}
NOTES: {"; ".join(notes) if notes else ""}

Return STRICT JSON only. Do not wrap in code fences.
""".strip()

    context_obj = {
        "variable": variable,
        "sections": sections,
        "subsections": subsections,
        "line_items": line_items,
        "total_words": total_words,
        "avoid_words": avoid_words,
        "units": units,
        "notes": notes,
        "pdf_path": pdf_path,
        "require_coordinates": require_coordinates,
        "top_k": return_top_k
    }
    return {
        "system": system_instructions,
        "user": user_message,
        "context_json": json.dumps(context_obj, ensure_ascii=False)
    }


# ---------------------------
# Image ↔ base64 helpers
# ---------------------------

def image_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def pdf_to_images(pdf_path: str, dpi: int = 200) -> List[Image.Image]:
    """
    Convert a PDF to a list of PIL Images (one per page).
    Requires poppler installed on system for pdf2image.
    """
    return convert_from_path(pdf_path, dpi=dpi)


# ---------------------------
# Gemini REST helpers
# ---------------------------

def build_request_body(
    *, prompt_parts: Dict[str, str], image_b64: str, model: str,
    temperature: float = 0.2, top_k: int = 32, top_p: float = 0.95, max_tokens: int = 2048
) -> Dict[str, Any]:
    """
    Build a request body compatible with Generative Language API generateContent.
    We provide system instructions both via 'system_instruction' and as the first part of the user message
    for broader compatibility.
    """
    system_text = prompt_parts["system"]
    user_text = prompt_parts["user"]
    context_json = prompt_parts["context_json"]

    body = {
        "system_instruction": {"role": "system", "parts": [{"text": system_text}]},
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": system_text},  # redundancy: some runtimes ignore system role
                    {"text": user_text},
                    {"text": context_json},
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": image_b64
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "topK": top_k,
            "topP": top_p,
            "maxOutputTokens": max_tokens
        }
    }
    return body


def gemini_generate(
    *, endpoint_tmpl: str, model: str, api_key: str, body: Dict[str, Any], timeout: int = TIMEOUT_SEC
) -> Dict[str, Any]:
    url = endpoint_tmpl.format(model=model)
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }
    resp = requests.post(url, headers=headers, json=body, timeout=timeout)
    if resp.status_code >= 400:
        raise RuntimeError(f"Gemini error {resp.status_code}: {resp.text}")
    return resp.json()


def extract_text_from_response(resp_json: Dict[str, Any]) -> str:
    """
    Pulls the first text part from 'candidates' → 'content' → 'parts'.
    """
    # Generative Language API typical structure:
    try:
        candidates = resp_json.get("candidates", [])
        for c in candidates:
            content = c.get("content") or {}
            parts = content.get("parts", [])
            for part in parts:
                if "text" in part:
                    return part["text"]
        # Some variants might deliver top-level 'text'
        if "text" in resp_json:
            return resp_json["text"]
    except Exception:
        pass
    return ""


def parse_strict_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Try to parse a JSON object from model output robustly:
    - strip code fences/backticks
    - take substring from first '{' to last '}' if needed
    """
    if not text:
        return None
    # Remove code fences/backticks
    stripped = re.sub(r"^```.*?\n|\n```$", "", text.strip(), flags=re.DOTALL)
    # Find JSON object bounds
    if "{" in stripped and "}" in stripped:
        start = stripped.find("{")
        end = stripped.rfind("}") + 1
        candidate = stripped[start:end]
    else:
        candidate = stripped
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Attempt to remove trailing commas or other minor issues
        candidate2 = re.sub(r",\s*}", "}", candidate)
        candidate2 = re.sub(r",\s*]", "]", candidate2)
        try:
            return json.loads(candidate2)
        except Exception:
            return None


# ---------------------------
# Page-level extraction
# ---------------------------

@dataclass
class PageResult:
    page_index: int           # 0-based
    response_text: str        # raw model text
    parsed: Optional[Dict[str, Any]]
    error: Optional[str] = None


def extract_from_image_page(
    *, image: Image.Image, variable: str, pdf_path: str, model: str, api_key: str,
    endpoint_tmpl: str, prefer_current_period: bool = True, return_top_k: int = 1
) -> PageResult:
    try:
        prompt_parts = build_schema_aware_prompt(
            pdf_path=pdf_path,
            variable=variable,
            cfg=VARIABLES_CFG,
            page_range=None,  # page is self-contained
            period_hint=None,
            prefer_current_period=prefer_current_period,
            return_top_k=return_top_k,
            require_coordinates=True,
        )
        img_b64 = image_to_base64(image, fmt="PNG")
        body = build_request_body(prompt_parts=prompt_parts, image_b64=img_b64, model=model)
        resp_json = gemini_generate(endpoint_tmpl=endpoint_tmpl, model=model, api_key=api_key, body=body)
        text = extract_text_from_response(resp_json)
        parsed = parse_strict_json(text)
        return PageResult(page_index=-1, response_text=text, parsed=parsed, error=None)
    except Exception as e:
        return PageResult(page_index=-1, response_text="", parsed=None, error=str(e))


# ---------------------------
# Orchestrators
# ---------------------------

def find_variable_in_pdf(
    pdf_path: str,
    variable: str,
    model: str = DEFAULT_MODEL,
    endpoint_tmpl: str = DEFAULT_ENDPOINT_TMPL,
    max_workers: int = MAX_WORKERS,
    dpi: int = 200,
    prefer_current_period: bool = True,
    return_top_k: int = 1
) -> Dict[str, Any]:
    """
    Process all pages of the PDF in parallel and return the best match.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set. Export your API key first.")

    images = pdf_to_images(pdf_path, dpi=dpi)
    results: List[PageResult] = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_to_idx = {}
        for idx, img in enumerate(images):
            fut = ex.submit(
                extract_from_image_page,
                image=img, variable=variable, pdf_path=pdf_path, model=model,
                api_key=api_key, endpoint_tmpl=endpoint_tmpl,
                prefer_current_period=prefer_current_period, return_top_k=return_top_k
            )
            fut_to_idx[fut] = idx

        for fut in as_completed(fut_to_idx):
            idx = fut_to_idx[fut]
            pr = fut.result()
            pr.page_index = idx
            results.append(pr)

    # Pick best: highest confidence among parsed found=true
    best = None
    best_conf = -1.0
    for pr in results:
        if pr.parsed and isinstance(pr.parsed, dict):
            found = pr.parsed.get("found", False)
            conf = float(pr.parsed.get("confidence", 0.0))
            if found and conf > best_conf:
                best = pr
                best_conf = conf

    return {
        "variable": variable,
        "pdf_path": pdf_path,
        "model": model,
        "endpoint": endpoint_tmpl.format(model=model),
        "pages": len(images),
        "results": [
            {
                "page_index": r.page_index,
                "parsed": r.parsed,
                "error": r.error
            } for r in sorted(results, key=lambda x: x.page_index)
        ],
        "best": None if best is None else {
            "page_index": best.page_index,
            "parsed": best.parsed
        }
    }


def find_multiple_variables_in_pdf(
    pdf_path: str,
    variables: List[str],
    **kwargs
) -> Dict[str, Any]:
    out = {"pdf_path": pdf_path, "variables": {}}
    for var in variables:
        out["variables"][var] = find_variable_in_pdf(pdf_path, var, **kwargs)
    return out


# ---------------------------
# CLI
# ---------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Schema-aware VLM extractor for fixed-format financial PDFs.")
    p.add_argument("pdf_path", help="Path to the PDF printed worksheet.")
    p.add_argument("--variable", action="append", required=True,
                   help="Variable to extract. Repeat flag for multiple variables.")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Gemini model (default: {DEFAULT_MODEL})")
    p.add_argument("--workers", type=int, default=MAX_WORKERS, help="Max parallel page workers.")
    p.add_argument("--dpi", type=int, default=200, help="PDF render DPI (200–300 recommended).")
    p.add_argument("--return-top-k", type=int, default=1, help="Return up to K alternates in disambiguation.")
    p.add_argument("--no-prefer-latest", action="store_true", help="Do NOT prefer latest period totals.")
    return p


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args()

    prefer_latest = not args.no_prefer_latest
    if len(args.variable) == 1:
        res = find_variable_in_pdf(
            args.pdf_path,
            args.variable[0],
            model=args.model,
            max_workers=args.workers,
            dpi=args.dpi,
            return_top_k=args.return_top_k,
            prefer_current_period=prefer_latest
        )
    else:
        res = find_multiple_variables_in_pdf(
            args.pdf_path,
            args.variable,
            model=args.model,
            max_workers=args.workers,
            dpi=args.dpi,
            return_top_k=args.return_top_k,
            prefer_current_period=prefer_latest
        )

    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

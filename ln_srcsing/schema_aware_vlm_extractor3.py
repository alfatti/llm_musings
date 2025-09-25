#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schema-aware PDF → JSON extractor using a VLM (Gemini via REST).

Features
- Converts PDF pages to images (pdf2image + poppler required)
- Builds hierarchy-aware prompts (sections → sub-sections/columns → line-items)
- Parallelizes per-page VLM calls (ThreadPoolExecutor)
- Returns STRICT JSON per page + best pick by model-reported confidence
- **Value returned exactly as printed (text)** — no normalization or rescaling
- Variant lists are **examples/hints**, not strict names (use fuzzy/semantic matching)
- Choose exactly which variable to extract (no implicit "all")

ENV
  GEMINI_API_KEY   : required
  GEMINI_ENDPOINT  : optional, default => https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
  GEMINI_MODEL     : optional, default => gemini-1.5-flash

USAGE (single variable)
  python schema_aware_vlm_extractor.py ./sheet.pdf --variable "Outstanding Loan Balance"

Explore / inspect
  python schema_aware_vlm_extractor.py --list-variables
  python schema_aware_vlm_extractor.py --show-variable "Cash-Interest"

Extend config at runtime
  python schema_aware_vlm_extractor.py ./sheet.pdf --variable "My Custom Var" --cfg ./vars.json
  # where vars.json is: { "My Custom Var": { "sections": [...], "subsections": [...], "line_items": [...], ... } }

Requires
  pip install pdf2image pillow requests
  and Poppler installed (for pdf2image)
"""
import os
import io
import re
import json
import base64
import logging
import argparse
import difflib
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
# Variable selection helpers
# ---------------------------

def available_variables() -> List[str]:
    return sorted(VARIABLES_CFG.keys())

def suggest_variable(name: str) -> List[str]:
    return difflib.get_close_matches(name, available_variables(), n=3, cutoff=0.5)

def get_variable_cfg(name: str) -> Dict[str, Any]:
    if name not in VARIABLES_CFG:
        sugg = suggest_variable(name)
        hint = f" Did you mean: {', '.join(sugg)}?" if sugg else ""
        raise KeyError(f"Unknown variable '{name}'. Available: {', '.join(available_variables())}.{hint}")
    return VARIABLES_CFG[name]

def extend_variables_cfg(user_cfg: Dict[str, Any]) -> None:
    # Shallow update per variable key; each entry should mirror the expected structure.
    for k, v in user_cfg.items():
        if isinstance(v, dict):
            if k in VARIABLES_CFG and isinstance(VARIABLES_CFG[k], dict):
                VARIABLES_CFG[k].update(v)
            else:
                VARIABLES_CFG[k] = v

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
    vc = get_variable_cfg(variable)

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

    schema_desc = {
        "variable": "string  # echo requested variable",
        "found": "boolean",
        "value": "string|null  # EXACT TEXT as printed for the value cell; do not alter or normalize",
        "unit": "string|null    # '$', 'USD', 'mm', 'k' (if printed near the value)",
        "scale_applied": "string|null  # e.g., 'in millions' if stated; report separately",
        "sign": "string|null     # keep as printed if part of the value text",
        "page_number": "integer|null  # 1-based page index if known",
        "bbox": "[x0,y0,x1,y1]|null   # bounding box for chosen value if available",
        "section_hit": "string|null    # exact observed label if available",
        "subsection_hit": "string|null",
        "line_item_hit": "string|null",
        "evidence_text": "string|null  # short local quote near the value",
        "confidence": "number  # 0..1",
        "disambiguation": [
            {
                "value": "string",
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
  5) Return the value EXACTLY AS PRINTED (as text). Do NOT alter spacing, punctuation, currency symbols, parentheses, or separators. Report any stated units/scale separately.

{page_hint}
{period_hint_text}
{period_rule}

Handle messy layouts: tables may be skewed, headers may be multi-line, totals may appear as 'Total', 'Aggregate', 'Sum'.
Variant lists for SECTION/SUB-SECTION/LINE-ITEM are examples/hints of how labels may vary; use fuzzy/semantic matching.
Do NOT force labels to match exactly and do NOT rename labels. Return the exact observed label text in *_hit fields when available.

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

VARIANT LISTS ARE EXAMPLES, NOT STRICT MATCH RULES.
Return the value EXACTLY AS PRINTED (text).
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
                    {"text": system_text},  # redundancy for runtimes that ignore system role
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
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    resp = requests.post(url, headers=headers, json=body, timeout=timeout)
    if resp.status_code >= 400:
        raise RuntimeError(f"Gemini error {resp.status_code}: {resp.text}")
    return resp.json()


def extract_text_from_response(resp_json: Dict[str, Any]) -> str:
    """
    Pulls the first text part from 'candidates' → 'content' → 'parts'.
    """
    try:
        candidates = resp_json.get("candidates", [])
        for c in candidates:
            content = c.get("content") or {}
            parts = content.get("parts", [])
            for part in parts:
                if "text" in part:
                    return part["text"]
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
    stripped = re.sub(r"^```.*?\n|\n```$", "", text.strip(), flags=re.DOTALL)
    if "{" in stripped and "}" in stripped:
        start = stripped.find("{")
        end = stripped.rfind("}") + 1
        candidate = stripped[start:end]
    else:
        candidate = stripped
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
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
            page_range=None,
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
    Extract **one chosen variable** from a multi-page PDF (printed spreadsheet).
    Parallelizes one VLM call per page, then picks the best by confidence.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set. Export your API key first.")

    # Validate variable choice early (and show suggestions on typos)
    _ = get_variable_cfg(variable)

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


def extract_variable(pdf_path: str, variable: str, **kwargs) -> Dict[str, Any]:
    """
    Programmatic helper (alias): extract one chosen variable from the PDF.
    """
    return find_variable_in_pdf(pdf_path, variable, **kwargs)


# ---------------------------
# CLI
# ---------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Schema-aware VLM extractor for fixed-format financial PDFs.")
    p.add_argument("pdf_path", nargs="?", help="Path to the PDF printed worksheet.")
    p.add_argument("--variable", action="append", required=False,
                   help="Variable to extract. Repeat flag for multiple variables (still runs one at a time).")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Gemini model (default: {DEFAULT_MODEL})")
    p.add_argument("--workers", type=int, default=MAX_WORKERS, help="Max parallel page workers.")
    p.add_argument("--dpi", type=int, default=200, help="PDF render DPI (200–300 recommended).")
    p.add_argument("--return-top-k", type=int, default=1, help="Return up to K alternates in disambiguation.")
    p.add_argument("--no-prefer-latest", action="store_true", help="Do NOT prefer latest period totals.")
    p.add_argument("--list-variables", action="store_true", help="List available variable keys and exit.")
    p.add_argument("--show-variable", type=str, help="Print the config for a single variable and exit.")
    p.add_argument("--cfg", type=str, help="Path to JSON file to extend/override VARIABLES_CFG.")
    return p


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args()

    # Info modes that don't require a pdf_path
    if args.list_variables:
        print("\n".join(available_variables()))
        return
    if args.show_variable:
        print(json.dumps(get_variable_cfg(args.show_variable), indent=2))
        return

    if not args.pdf_path:
        parser.error("Missing pdf_path (or use --list-variables / --show-variable).")

    # Load external CFG if provided
    if args.cfg:
        with open(args.cfg, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        if not isinstance(user_cfg, dict):
            raise TypeError("Config file must be a JSON object mapping variable -> config dict")
        extend_variables_cfg(user_cfg)

    if not args.variable:
        parser.error("Please provide --variable NAME (or use --list-variables to see choices).")

    prefer_latest = not args.no_prefer_latest

    # Even if multiple --variable flags are provided, run them one-by-one (no 'all at once')
    all_results = {}
    for var in args.variable:
        res = find_variable_in_pdf(
            args.pdf_path,
            var,
            model=args.model,
            max_workers=args.workers,
            dpi=args.dpi,
            return_top_k=args.return_top_k,
            prefer_current_period=prefer_latest
        )
        all_results[var] = res

    # If only one variable was requested, print that object; otherwise a dict keyed by variable
    if len(all_results) == 1:
        print(json.dumps(next(iter(all_results.values())), indent=2))
    else:
        print(json.dumps({"pdf_path": args.pdf_path, "variables": all_results}, indent=2))


if __name__ == "__main__":
    main()

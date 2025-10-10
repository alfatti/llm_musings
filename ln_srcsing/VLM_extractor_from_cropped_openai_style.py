#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VLM extraction (OpenAI-style -> Gemini) for Basel risk inputs from a 1-page PDF.

Features:
- Apigee/semantic headers merged with your BASE_HEADERS
- One-page PDF -> JPEG (dpi configurable), embedded as data URL
- Prompt uses name-pointer hints (section, sub-section, line-item) + prior cell vicinity (M66, etc.)
- JSON-only output with schema + robust parser
- Designed for "elation" sanity tests: swap variables & hints easily

Usage:
  python vlm_bazel_risk_extract.py \
    --pdf path/to/page.pdf \
    --variable "Risk Weight" \
    --section-variants "Risk Inputs,Risk Factors" \
    --subsection-variants "Portfolio A,Loans" \
    --line-item-variants "Risk Weight,RW,RWA Weight" \
    --cell-hint "M66" \
    --vicinity-rows 6 \
    --vicinity-cols 2 \
    --dpi 350

Environment (preferred) or CLI flags:
  WF_API_KEY, WF_USECASE_ID, WF_APP_ID must be set (or pass --api-key, --usecase-id, --app-id)
"""

import argparse
import base64
import datetime as dt
import json
import os
import re
import string
import uuid
from io import BytesIO

from pdf2image import convert_from_path
from PIL import Image

# --------- Utilities ---------

def generate_uuid() -> str:
    return str(uuid.uuid4())

def iso_now_utc() -> str:
    # Example format: 2025-10-10T14:05:22Z
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def col_letter_to_index(col: str) -> int:
    """Convert Excel column letters to 1-based index (A->1, Z->26, AA->27, ...)."""
    col = col.upper().strip()
    idx = 0
    for c in col:
        if c in string.ascii_uppercase:
            idx = idx * 26 + (ord(c) - ord('A') + 1)
    return idx

def parse_cell(cell_ref: str):
    """Split a cell like 'M66' -> ('M', 66, col_index=13). If not valid, returns (None, None, None)."""
    if not cell_ref:
        return None, None, None
    m = re.match(r"^\s*([A-Za-z]+)\s*([0-9]+)\s*$", cell_ref)
    if not m:
        return None, None, None
    col_letters, row_str = m.group(1), m.group(2)
    col_idx = col_letter_to_index(col_letters)
    return col_letters.upper(), int(row_str), col_idx

def pdf_to_data_url(pdf_path: str, dpi: int = 350, jpeg_quality: int = 92) -> str:
    """
    Convert a 1-page PDF to JPEG base64 data URL.
    dpi ~300-400 generally works well for small/financial fonts; default 350.
    """
    pages = convert_from_path(pdf_path, dpi=dpi)
    if not pages:
        raise ValueError("No pages found in PDF.")
    if len(pages) > 1:
        # If multi-page slips through, just take the first page for this 'elation' test
        page = pages[0]
    else:
        page = pages[0]
    # Convert to JPEG in-memory
    buf = BytesIO()
    page.convert("RGB").save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def coerce_json_from_text(text: str):
    """
    Try hard to extract the first valid JSON object from the model output.
    - Strips code fences if present
    - Finds the outermost {...}
    - Fixes a few common mistakes (trailing commas, smart quotes)
    """
    if text is None:
        raise ValueError("Empty completion text.")

    # Remove code fences if any
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "").strip()

    # Replace smart quotes
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("`", '"')

    # Extract the first {...} block
    brace_stack = []
    start_idx = None
    for i, ch in enumerate(text):
        if ch == "{":
            if start_idx is None:
                start_idx = i
            brace_stack.append("{")
        elif ch == "}":
            if brace_stack:
                brace_stack.pop()
                if not brace_stack and start_idx is not None:
                    candidate = text[start_idx:i + 1]
                    # Try parsing candidate; if fails, attempt minor repairs
                    try:
                        return json.loads(candidate)
                    except Exception:
                        repaired = repair_minor_json_issues(candidate)
                        return json.loads(repaired)
    # Fallback: try entire string
    try:
        return json.loads(text)
    except Exception:
        repaired = repair_minor_json_issues(text)
        return json.loads(repaired)

def repair_minor_json_issues(s: str) -> str:
    """
    Very conservative minor repairs:
    - Remove trailing commas before } or ]
    - Ensure keys are quoted with double-quotes (best-effort)
    """
    # Remove trailing commas: "..., }" or "..., ]"
    s = re.sub(r",\s*([}\]])", r"\1", s)

    # Best-effort key quoting: convert single-quoted keys to double-quoted (limited)
    # WARNING: conservative to avoid mangling values
    s = re.sub(r"(\{|,)\s*'([^']+)'\s*:", r'\1 "\2":', s)

    # Replace single-quoted string values with double-quoted (limited)
    s = re.sub(r":\s*'([^']*)'\s*(,|\})", r': "\1"\2', s)

    return s

# --------- Prompt builder ---------

def build_prompt(
    variable_name: str,
    section_variants: list[str],
    subsection_variants: list[str],
    line_item_variants: list[str],
    cell_hint: str | None,
    vicinity_rows: int,
    vicinity_cols: int,
):
    col_letters, hint_row, hint_col_idx = parse_cell(cell_hint) if cell_hint else (None, None, None)

    vicinity_desc = {
        "bias": "vertical-first",
        "rows_to_check_above": vicinity_rows,
        "rows_to_check_below": vicinity_rows,
        "cols_to_check_left": vicinity_cols,
        "cols_to_check_right": vicinity_cols,
        "cell_hint": cell_hint or "UNKNOWN",
        "parsed_hint": {
            "column_letters": col_letters,
            "row_index": hint_row,
            "column_index_1_based": hint_col_idx,
        },
    }

    schema = {
        "type": "object",
        "required": ["variable", "value_text", "evidence", "confidence"],
        "properties": {
            "variable": {"type": "string", "description": "Exact variable you extracted."},
            "value_text": {"type": "string", "description": "Value exactly as it appears in the sheet."},
            "numeric_value": {"type": ["number", "null"], "description": "If value is numeric, parsed float; else null."},
            "unit": {"type": ["string", "null"], "description": "Unit if present (e.g., %, bps, $), else null."},
            "detected_cell": {"type": ["string", "null"], "description": "Detected Excel-like cell (e.g., N67) if resolvable."},
            "evidence": {
                "type": "object",
                "properties": {
                    "section_match": {"type": ["string", "null"]},
                    "subsection_match": {"type": ["string", "null"]},
                    "line_item_match": {"type": ["string", "null"]},
                    "vicinity_strategy": {"type": "object"},
                    "notes": {"type": ["string", "null"]},
                },
                "required": ["vicinity_strategy"],
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
    }

    # JSON-only, zero-shot prompt with deterministic instructions
    sys = (
        "You are a precise financial document vision assistant. "
        "You inspect a single-page image of a spreadsheet exported to PDF and extract one requested variable. "
        "Return STRICT JSON ONLY with the provided schema. Do not include markdown or explanations."
    )

    user_text = {
        "task": f"Extract the variable: '{variable_name}'.",
        "how_to_find_it": {
            "name_pointers": {
                "section_variants": section_variants,
                "subsection_variants": subsection_variants,
                "line_item_variants": line_item_variants,
                "notes": (
                    "Variants represent how the labels may appear; use fuzzy matching and typography cues. "
                    "If 'Section' is actually a column header in this layout, treat it as such."
                ),
            },
            "cell_hint_and_vicinity": vicinity_desc,
            "layout_prior": (
                "Rows typically correspond to loans and can shift between report versions. "
                "Prioritize vertical scanning near the hint row; if not found, expand left/right within the given column window."
            ),
            "value_rules": [
                "Return the value exactly as text from the sheet (no normalization).",
                "If there are footnotes or superscripts, exclude them from value_text but mention in evidence.notes.",
            ],
            "tie_breakers": [
                "Prefer a match where section/sub-section/line-item cues align best.",
                "If multiple candidates, choose the one nearest to the hint row.",
            ],
        },
        "output_schema": schema,
        "formatting": "Return STRICT JSON ONLY (no markdown code fences, no backticks).",
    }

    return sys, user_text

# --------- Core call ---------

def call_gemini_openai_style(
    openai_client,
    model: str,
    data_url: str,
    system_text: str,
    user_payload: dict,
    headers: dict,
    temperature: float = 0.0,
    max_tokens: int = 800,
):
    """
    Calls Gemini via OpenAI-style chat.completions API with image + text.
    We pass custom headers via extra_headers.
    """
    messages = [
        {"role": "system", "content": system_text},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": json.dumps(user_payload, ensure_ascii=False)},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]

    resp = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_headers=headers,  # <-- important
    )
    # Expect OpenAI-style: choices[0].message.content
    content = resp.choices[0].message.content if resp and resp.choices else None
    return content, resp

# --------- Header assembly ---------

def assemble_headers(
    api_key: str,
    usecase_id: str,
    app_id: str,
    apigee_access_token: str,
):
    BASE_HEADERS = {
        "x-wf-api-key": api_key,
        "x-wf-usecase-id": usecase_id,
        "x-wf-client-id": app_id,
    }

    request_id = correlation_id = generate_uuid()
    request_date = iso_now_utc()

    generate_semantic_headers = BASE_HEADERS | {
        "x-request-id": request_id,
        "x-wf-request-date": request_date,
        "x-correlation-id": correlation_id,
        "Content-Type": "application/json",
        "Authorization": f"Bearer {apigee_access_token}",
    }
    return generate_semantic_headers

# --------- Main flow ---------

def run_extraction(
    openai_client,
    pdf_path: str,
    variable_name: str,
    section_variants: list[str],
    subsection_variants: list[str],
    line_item_variants: list[str],
    cell_hint: str | None,
    vicinity_rows: int,
    vicinity_cols: int,
    dpi: int,
    api_key: str,
    usecase_id: str,
    app_id: str,
    apigee_access_token: str,
    model: str = "gemini-2.5.pro",
):
    # 1) PDF -> data URL
    data_url = pdf_to_data_url(pdf_path, dpi=dpi)

    # 2) Build prompt
    system_text, user_payload = build_prompt(
        variable_name=variable_name,
        section_variants=section_variants,
        subsection_variants=subsection_variants,
        line_item_variants=line_item_variants,
        cell_hint=cell_hint,
        vicinity_rows=vicinity_rows,
        vicinity_cols=vicinity_cols,
    )

    # 3) Headers
    headers = assemble_headers(
        api_key=api_key,
        usecase_id=usecase_id,
        app_id=app_id,
        apigee_access_token=apigee_access_token,
    )

    # 4) Call model
    raw_text, raw_resp = call_gemini_openai_style(
        openai_client=openai_client,
        model=model,
        data_url=data_url,
        system_text=system_text,
        user_payload=user_payload,
        headers=headers,
        temperature=0.0,
        max_tokens=800,
    )

    # 5) Parse JSON (robust)
    parsed = coerce_json_from_text(raw_text)

    return {
        "raw_text": raw_text,
        "parsed": parsed,
        "request_headers_used": {k: headers[k] for k in headers if k not in ("Authorization",)},  # mask token
    }

# --------- CLI ---------

def parse_list_arg(csv_or_repeatable: list[str]) -> list[str]:
    # Accept either a single comma-separated value or multiple flags
    if not csv_or_repeatable:
        return []
    if len(csv_or_repeatable) == 1 and ("," in csv_or_repeatable[0]):
        return [x.strip() for x in csv_or_repeatable[0].split(",") if x.strip()]
    return [x.strip() for x in csv_or_repeatable if x.strip()]

def main():
    parser = argparse.ArgumentParser(description="VLM Basel risk input extractor (OpenAI-style -> Gemini).")
    parser.add_argument("--pdf", required=True, help="Path to 1-page PDF exported from Excel.")
    parser.add_argument("--variable", required=True, help="Name of the variable to extract (e.g., 'Risk Weight').")
    parser.add_argument("--section-variants", nargs="+", default=[], help="List or comma-separated string.")
    parser.add_argument("--subsection-variants", nargs="+", default=[], help="List or comma-separated string.")
    parser.add_argument("--line-item-variants", nargs="+", default=[], help="List or comma-separated string.")
    parser.add_argument("--cell-hint", default=None, help="Excel-like cell hint (e.g., 'M66').")
    parser.add_argument("--vicinity-rows", type=int, default=6, help="Rows above/below the hint to scan.")
    parser.add_argument("--vicinity-cols", type=int, default=2, help="Columns left/right of the hint to scan.")
    parser.add_argument("--dpi", type=int, default=350, help="JPEG DPI for PDF rasterization (300–400 recommended).")
    parser.add_argument("--model", default="gemini-2.5.pro", help="Model name.")
    # Credentials (env fallbacks)
    parser.add_argument("--api-key", default=os.getenv("WF_API_KEY"), help="x-wf-api-key")
    parser.add_argument("--usecase-id", default=os.getenv("WF_USECASE_ID"), help="x-wf-usecase-id")
    parser.add_argument("--app-id", default=os.getenv("WF_APP_ID"), help="x-wf-client-id")
    parser.add_argument("--apigee-access-token", default=os.getenv("APIGEE_ACCESS_TOKEN"), help="Bearer token")

    args = parser.parse_args()

    missing = []
    if not args.api_key: missing.append("WF_API_KEY / --api-key")
    if not args.usecase_id: missing.append("WF_USECASE_ID / --usecase-id")
    if not args.app_id: missing.append("WF_APP_ID / --app-id")
    if not args.apigee_access_token: missing.append("APIGEE_ACCESS_TOKEN / --apigee-access-token")
    if missing:
        raise SystemExit(f"Missing credentials: {', '.join(missing)}")

    # Defer import of your openai_client to whatever your environment provides.
    # Example:
    #   from your_package.clients import openai_client
    try:
        from openai import OpenAI
        # If you already have an instantiated client elsewhere, replace this with your import.
        # This OpenAI() construction is a placeholder – your runtime should supply a configured client.
        openai_client = OpenAI()  # In your env, this is typically already configured.
    except Exception:
        raise SystemExit(
            "Could not import/construct `openai_client`. Ensure your environment provides it, or edit this file to import your own."
        )

    result = run_extraction(
        openai_client=openai_client,
        pdf_path=args.pdf,
        variable_name=args.variable,
        section_variants=parse_list_arg(args.section_variants),
        subsection_variants=parse_list_arg(args.subsection_variants),
        line_item_variants=parse_list_arg(args.line_item_variants),
        cell_hint=args.cell_hint,
        vicinity_rows=args.vicinity_rows,
        vicinity_cols=args.vicinity_cols,
        dpi=args.dpi,
        api_key=args.api_key,
        usecase_id=args.usecase_id,
        app_id=args.app_id,
        apigee_access_token=args.apigee_access_token,
        model=args.model,
    )

    print("\n=== RAW MODEL TEXT ===\n")
    print(result["raw_text"])
    print("\n=== PARSED JSON ===\n")
    print(json.dumps(result["parsed"], indent=2, ensure_ascii=False))
    print("\n=== REQUEST HEADER SUMMARY (sans token) ===\n")
    print(json.dumps(result["request_headers_used"], indent=2))

if __name__ == "__main__":
    main()

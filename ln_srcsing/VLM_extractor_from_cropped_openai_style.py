#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VLM extraction (OpenAI-style -> Gemini via Apigee proxy) for Basel risk inputs from a 1-page PDF.
Python 3.8/3.9 compatible (no PEP 604 unions, no dict | merges).

Usage:
  python vlm_bazel_risk_extract_py39.py \
    --pdf path/to/page.pdf \
    --variable "Risk Weight" \
    --section-variants "Risk Inputs,Risk Factors" \
    --subsection-variants "Portfolio A,Loans" \
    --line-item-variants "Risk Weight,RW,RWA Weight" \
    --cell-hint "M66" \
    --vicinity-rows 6 \
    --vicinity-cols 2 \
    --dpi 350 \
    --apigee-base-url "https://company-gateway.apigee.net/your/path/v1/openai/v1"

Env vars (preferred) or CLI flags:
  WF_API_KEY / --api-key
  WF_USECASE_ID / --usecase-id
  WF_APP_ID / --app-id
  APIGEE_ACCESS_TOKEN / --apigee-access-token
  APIGEE_BASE_URL / --apigee-base-url
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
from typing import Optional, List, Tuple, Dict, Any

from pdf2image import convert_from_path
from PIL import Image  # noqa: F401  (used indirectly via pdf2image PIL.Image)

# --------- Utilities ---------

def generate_uuid() -> str:
    return str(uuid.uuid4())

def iso_now_utc() -> str:
    # Example: 2025-10-10T14:05:22Z
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def col_letter_to_index(col: str) -> int:
    """Excel column letters -> 1-based index (A->1, Z->26, AA->27, ...)."""
    col = col.upper().strip()
    idx = 0
    for c in col:
        if "A" <= c <= "Z":
            idx = idx * 26 + (ord(c) - ord('A') + 1)
    return idx

def parse_cell(cell_ref: str) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    """'M66' -> ('M', 66, 13). If invalid, returns (None, None, None)."""
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
    Convert a 1-page PDF to a JPEG data URL. 300–400 DPI is usually best for financial fonts.
    """
    pages = convert_from_path(pdf_path, dpi=dpi)
    if not pages:
        raise ValueError("No pages found in PDF.")
    page = pages[0]
    buf = BytesIO()
    page.convert("RGB").save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return "data:image/jpeg;base64," + b64

def coerce_json_from_text(text: Optional[str]) -> Dict[str, Any]:
    """
    Extract the first valid JSON object from model output.
    - Strips code fences
    - Replaces smart quotes
    - Repairs a few minor issues (trailing commas, single-quoted keys/values)
    """
    if text is None:
        raise ValueError("Empty completion text.")

    # Remove code fences
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "").strip()

    # Replace smart quotes & backticks
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("`", '"')

    brace_stack: List[str] = []
    start_idx: Optional[int] = None
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
                    try:
                        return json.loads(candidate)
                    except Exception:
                        repaired = repair_minor_json_issues(candidate)
                        return json.loads(repaired)

    # Fallback: whole string
    try:
        return json.loads(text)
    except Exception:
        repaired = repair_minor_json_issues(text)
        return json.loads(repaired)

def repair_minor_json_issues(s: str) -> str:
    """
    Conservative repairs:
    - Remove trailing commas before } or ]
    - Convert single-quoted keys/values to double-quoted (best-effort)
    """
    s = re.sub(r",\s*([}\]])", r"\1", s)
    s = re.sub(r"(\{|,)\s*'([^']+)'\s*:", r'\1 "\2":', s)
    s = re.sub(r':\s*\'([^\']*)\'\s*(,|\})', r': "\1"\2', s)
    return s

# --------- Prompt builder ---------

def build_prompt(
    variable_name: str,
    section_variants: List[str],
    subsection_variants: List[str],
    line_item_variants: List[str],
    cell_hint: Optional[str],
    vicinity_rows: int,
    vicinity_cols: int,
) -> Tuple[str, Dict[str, Any]]:
    col_letters, hint_row, hint_col_idx = parse_cell(cell_hint) if cell_hint else (None, None, None)

    vicinity_desc: Dict[str, Any] = {
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

    schema: Dict[str, Any] = {
        "type": "object",
        "required": ["variable", "value_text", "evidence", "confidence"],
        "properties": {
            "variable": {"type": "string", "description": "Exact variable you extracted."},
            "value_text": {"type": "string", "description": "Verbatim value as seen in the sheet."},
            "numeric_value": {"type": ["number", "null"], "description": "Parsed float if numeric, else null."},
            "unit": {"type": ["string", "null"], "description": "Unit (e.g., %, bps, $) if present."},
            "detected_cell": {"type": ["string", "null"], "description": "Excel-like cell (e.g., N67) if resolvable."},
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

    system_text: str = (
        "You are a precise financial document vision assistant. "
        "You inspect a single-page image of a spreadsheet exported to PDF and extract one requested variable. "
        "Return STRICT JSON ONLY with the provided schema. Do not include markdown or explanations."
    )

    user_text: Dict[str, Any] = {
        "task": "Extract the variable: '{}'.".format(variable_name),
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

    return system_text, user_text

# --------- OpenAI-style call via Apigee ---------

def call_gemini_openai_style(
    openai_client: Any,
    model: str,
    data_url: str,
    system_text: str,
    user_payload: Dict[str, Any],
    headers: Dict[str, str],
    temperature: float = 0.0,
    max_tokens: int = 800,
) -> Tuple[Optional[str], Any]:
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
        extra_headers=headers,  # no Authorization here; SDK supplies it
    )
    content = resp.choices[0].message.content if (resp and getattr(resp, "choices", None)) else None
    return content, resp

# --------- Header assembly (no Authorization) ---------

def assemble_headers(
    api_key: str,
    usecase_id: str,
    app_id: str,
) -> Dict[str, str]:
    base_headers = {
        "x-wf-api-key": api_key,
        "x-wf-usecase-id": usecase_id,
        "x-wf-client-id": app_id,
    }
    request_id = correlation_id = generate_uuid()
    request_date = iso_now_utc()
    merged = {
        **base_headers,
        "x-request-id": request_id,
        "x-wf-request-date": request_date,
        "x-correlation-id": correlation_id,
        "Content-Type": "application/json",
    }
    return merged

# --------- Orchestration ---------

def run_extraction(
    openai_client: Any,
    pdf_path: str,
    variable_name: str,
    section_variants: List[str],
    subsection_variants: List[str],
    line_item_variants: List[str],
    cell_hint: Optional[str],
    vicinity_rows: int,
    vicinity_cols: int,
    dpi: int,
    api_key: str,
    usecase_id: str,
    app_id: str,
    model: str = "gemini-2.5.pro",
) -> Dict[str, Any]:
    data_url = pdf_to_data_url(pdf_path, dpi=dpi)
    system_text, user_payload = build_prompt(
        variable_name=variable_name,
        section_variants=section_variants,
        subsection_variants=subsection_variants,
        line_item_variants=line_item_variants,
        cell_hint=cell_hint,
        vicinity_rows=vicinity_rows,
        vicinity_cols=vicinity_cols,
    )
    headers = assemble_headers(api_key=api_key, usecase_id=usecase_id, app_id=app_id)
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
    parsed = coerce_json_from_text(raw_text)
    return {
        "raw_text": raw_text,
        "parsed": parsed,
        "request_headers_used": headers,  # safe (no bearer)
    }

# --------- CLI ---------

def parse_list_arg(csv_or_repeatable: List[str]) -> List[str]:
    if not csv_or_repeatable:
        return []
    if len(csv_or_repeatable) == 1 and ("," in csv_or_repeatable[0]):
        return [x.strip() for x in csv_or_repeatable[0].split(",") if x.strip()]
    return [x.strip() for x in csv_or_repeatable if x.strip()]

def main():
    parser = argparse.ArgumentParser(description="VLM Basel risk input extractor (OpenAI-style -> Gemini via Apigee).")
    parser.add_argument("--pdf", required=True, help="Path to 1-page PDF exported from Excel.")
    parser.add_argument("--variable", required=True, help="Variable to extract, e.g., 'Risk Weight'.")
    parser.add_argument("--section-variants", nargs="+", default=[], help="List or comma-separated string.")
    parser.add_argument("--subsection-variants", nargs="+", default=[], help="List or comma-separated string.")
    parser.add_argument("--line-item-variants", nargs="+", default=[], help="List or comma-separated string.")
    parser.add_argument("--cell-hint", default=None, help="Excel-like cell hint (e.g., 'M66').")
    parser.add_argument("--vicinity-rows", type=int, default=6, help="Rows above/below the hint to scan.")
    parser.add_argument("--vicinity-cols", type=int, default=2, help="Columns left/right of the hint to scan.")
    parser.add_argument("--dpi", type=int, default=350, help="JPEG DPI for PDF rasterization (300–400 recommended).")
    parser.add_argument("--model", default="gemini-2.5.pro", help="Model name.")

    # Credentials & routing
    parser.add_argument("--api-key", default=os.getenv("WF_API_KEY"), help="x-wf-api-key")
    parser.add_argument("--usecase-id", default=os.getenv("WF_USECASE_ID"), help="x-wf-usecase-id")
    parser.add_argument("--app-id", default=os.getenv("WF_APP_ID"), help="x-wf-client-id")
    parser.add_argument("--apigee-access-token", default=os.getenv("APIGEE_ACCESS_TOKEN"), help="Bearer token for Apigee")
    parser.add_argument("--apigee-base-url", default=os.getenv("APIGEE_BASE_URL"), help="Apigee base URL for OpenAI proxy")

    args = parser.parse_args()

    missing = []
    if not args.api_key: missing.append("WF_API_KEY / --api-key")
    if not args.usecase_id: missing.append("WF_USECASE_ID / --usecase-id")
    if not args.app_id: missing.append("WF_APP_ID / --app-id")
    if not args.apigee_access_token: missing.append("APIGEE_ACCESS_TOKEN / --apigee-access-token")
    if not args.apigee_base_url: missing.append("APIGEE_BASE_URL / --apigee-base-url")
    if missing:
        raise SystemExit("Missing: " + ", ".join(missing))

    # Construct OpenAI client with Apigee proxy routing
    try:
        from openai import OpenAI
        openai_client = OpenAI(
            api_key=args.apigee_access_token,
            base_url=args.apigee_base_url,
        )
    except Exception as e:
        raise SystemExit("Failed to construct OpenAI client with Apigee routing: {}".format(e))

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
        model=args.model,
    )

    print("\n=== RAW MODEL TEXT ===\n")
    print(result["raw_text"])
    print("\n=== PARSED JSON ===\n")
    print(json.dumps(result["parsed"], indent=2, ensure_ascii=False))
    print("\n=== REQUEST HEADER SUMMARY (no bearer) ===\n")
    print(json.dumps(result["request_headers_used"], indent=2))

if __name__ == "__main__":
    main()

####################################################################
#FIXES
#####################################################################
#1) Replace your JSON helpers with these
import json, re
from typing import Optional, Dict, Any, List

def _strip_code_fences(text: str) -> str:
    # remove ```json ... ``` or ``` ... ``` fences
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())
    return text

def _find_fenced_json(text: str) -> Optional[str]:
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r"```\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if m:
        return m.group(1)
    return None

def _extract_first_json_object(text: str) -> Optional[str]:
    """
    Scan char-by-char, track string/escape/braces, return first balanced {...}.
    Robust to stray braces inside quoted strings.
    """
    in_string = False
    escape = False
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if start is None:
            if ch == "{":
                start = i
                depth = 1
            continue

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i+1]
        # continue scanning
    return None

def _repair_minimal(s: str) -> str:
    # 1) remove trailing commas before } or ]
    s = re.sub(r",\s*([}\]])", r"\1", s)
    # 2) normalize smart quotes to ASCII
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    # 3) convert single-quoted keys to double-quoted (best-effort)
    s = re.sub(r'(\{|,)\s*\'([^\'"]+)\'\s*:', r'\1 "\2":', s)
    # 4) convert single-quoted string values to double-quoted (best-effort)
    s = re.sub(r':\s*\'([^\'"]*)\'\s*([,}])', r': "\1"\2', s)
    return s

def _last_resort_cleanup(s: str) -> str:
    # Sometimes models inject stray backticks or unmatched quotes.
    s = s.replace("`", "")
    # Collapse CRLFs inside strings-ish contexts (very conservative)
    s = re.sub(r'"\s*\n\s*"', '" "', s)
    return s

def coerce_json_from_text(text: Optional[str]) -> Dict[str, Any]:
    if text is None:
        raise ValueError("Empty completion text.")

    # 0) direct fenced capture first
    fenced = _find_fenced_json(text)
    if fenced:
        try:
            return json.loads(fenced)
        except Exception:
            pass  # fall through

    # 1) strip fences and try raw
    raw = _strip_code_fences(text)
    try:
        return json.loads(raw)
    except Exception:
        pass

    # 2) extract first balanced {...}
    candidate = _extract_first_json_object(raw)
    if candidate:
        # 2a) try raw candidate
        try:
            return json.loads(candidate)
        except Exception:
            # 2b) try minimal repairs
            repaired = _repair_minimal(candidate)
            try:
                return json.loads(repaired)
            except Exception:
                # 2c) last resort cleanup
                repaired2 = _last_resort_cleanup(repaired)
                return json.loads(repaired2)

    # 3) as a final fallback, apply repairs to whole text
    repaired = _repair_minimal(raw)
    try:
        return json.loads(repaired)
    except Exception:
        repaired2 = _last_resort_cleanup(repaired)
        return json.loads(repaired2)

##################################################################
# 2) (Optional) Nudge the API to return JSON
# In your call_gemini_openai_style(...), change the call to:
##################################################################
import json, re
from typing import Optional, Dict, Any, List

def _strip_code_fences(text: str) -> str:
    # remove ```json ... ``` or ``` ... ``` fences
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())
    return text

def _find_fenced_json(text: str) -> Optional[str]:
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r"```\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if m:
        return m.group(1)
    return None

def _extract_first_json_object(text: str) -> Optional[str]:
    """
    Scan char-by-char, track string/escape/braces, return first balanced {...}.
    Robust to stray braces inside quoted strings.
    """
    in_string = False
    escape = False
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if start is None:
            if ch == "{":
                start = i
                depth = 1
            continue

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i+1]
        # continue scanning
    return None

def _repair_minimal(s: str) -> str:
    # 1) remove trailing commas before } or ]
    s = re.sub(r",\s*([}\]])", r"\1", s)
    # 2) normalize smart quotes to ASCII
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    # 3) convert single-quoted keys to double-quoted (best-effort)
    s = re.sub(r'(\{|,)\s*\'([^\'"]+)\'\s*:', r'\1 "\2":', s)
    # 4) convert single-quoted string values to double-quoted (best-effort)
    s = re.sub(r':\s*\'([^\'"]*)\'\s*([,}])', r': "\1"\2', s)
    return s

def _last_resort_cleanup(s: str) -> str:
    # Sometimes models inject stray backticks or unmatched quotes.
    s = s.replace("`", "")
    # Collapse CRLFs inside strings-ish contexts (very conservative)
    s = re.sub(r'"\s*\n\s*"', '" "', s)
    return s

def coerce_json_from_text(text: Optional[str]) -> Dict[str, Any]:
    if text is None:
        raise ValueError("Empty completion text.")

    # 0) direct fenced capture first
    fenced = _find_fenced_json(text)
    if fenced:
        try:
            return json.loads(fenced)
        except Exception:
            pass  # fall through

    # 1) strip fences and try raw
    raw = _strip_code_fences(text)
    try:
        return json.loads(raw)
    except Exception:
        pass

    # 2) extract first balanced {...}
    candidate = _extract_first_json_object(raw)
    if candidate:
        # 2a) try raw candidate
        try:
            return json.loads(candidate)
        except Exception:
            # 2b) try minimal repairs
            repaired = _repair_minimal(candidate)
            try:
                return json.loads(repaired)
            except Exception:
                # 2c) last resort cleanup
                repaired2 = _last_resort_cleanup(repaired)
                return json.loads(repaired2)

    # 3) as a final fallback, apply repairs to whole text
    repaired = _repair_minimal(raw)
    try:
        return json.loads(repaired)
    except Exception:
        repaired2 = _last_resort_cleanup(repaired)
        return json.loads(repaired2)


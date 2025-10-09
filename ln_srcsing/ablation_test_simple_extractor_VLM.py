#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basel Risk Input Extractor (single variable, one-page PDF -> Gemini 2.5 Pro via Apigee)

Features
- PDF -> image (configurable DPI; default 300) for better VLM OCR/reading.
- Prompt constructed from:
   * variable_name (target),
   * pointers: Section (often a column header), Sub-section (vertical cue), Line-item (horizontal cue),
   * previous_cell_hint (e.g., "M66") + "vicinity" window (vertical bias).
- Strict JSON schema requested from the model; robust JSON parsing with recovery.
- Apigee: uses an existing access token plus consumer key/secret headers.

Requirements
    pip install pdf2image pillow pydantic requests
Plus: poppler must be available on PATH for pdf2image.
"""

import base64
import io
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests
from pdf2image import convert_from_path
from PIL import Image
from pydantic import BaseModel, ValidationError, Field


# ---------------------------
# Models / Validation
# ---------------------------

class Coordinates(BaseModel):
    x: Optional[float] = None
    y: Optional[float] = None
    w: Optional[float] = None
    h: Optional[float] = None
    # Coordinates are optional; model may provide rough pixel bbox on the page image.

class Evidence(BaseModel):
    section: Optional[str] = None
    sub_section: Optional[str] = None
    line_item: Optional[str] = None
    cell_text_snapshot: Optional[str] = None
    nearest_cell_guess: Optional[str] = None
    coordinates: Optional[Coordinates] = None

class ExtractionResult(BaseModel):
    variable: str
    value_text: Optional[str] = None
    value_numeric: Optional[float] = None
    unit: Optional[str] = None
    found: bool
    evidence: Optional[Evidence] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    notes: Optional[str] = None


# ---------------------------
# PDF -> Image (Base64)
# ---------------------------

def pdf_to_base64_image(pdf_path: str, dpi: int = 300, max_width: Optional[int] = 2200) -> str:
    """
    Convert a SINGLE-page PDF to a base64-encoded PNG. Default DPI=300 (good balance for VLM).
    Optionally downscale to max_width to avoid overly huge payloads.
    """
    images = convert_from_path(pdf_path, dpi=dpi)
    if not images:
        raise ValueError("No pages found in PDF.")
    if len(images) > 1:
        # We proceed with the first page but warn.
        print("[WARN] PDF has multiple pages; using the first page only for this extractor.", file=sys.stderr)
    img: Image.Image = images[0].convert("RGB")

    if max_width and img.width > max_width:
        ratio = max_width / float(img.width)
        new_size = (max_width, int(img.height * ratio))
        img = img.resize(new_size, resample=Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# ---------------------------
# Excel cell helpers
# ---------------------------

_COLS = {chr(c): (c - 64) for c in range(65, 91)}  # 'A'->1 ... 'Z'->26

def excel_col_to_num(col: str) -> int:
    col = col.upper().strip()
    total = 0
    for ch in col:
        if ch < 'A' or ch > 'Z':
            raise ValueError(f"Invalid column letter: {col}")
        total = total * 26 + _COLS[ch]
    return total

def excel_cell_to_rc(cell: str) -> Dict[str, int]:
    """
    Convert Excel cell like 'M66' -> {'row': 66, 'col': 13}
    """
    m = re.fullmatch(r"\s*([A-Za-z]+)\s*([0-9]+)\s*", cell)
    if not m:
        raise ValueError(f"Invalid Excel cell reference: {cell}")
    col_letters, row_str = m.group(1), m.group(2)
    return {"row": int(row_str), "col": excel_col_to_num(col_letters)}


# ---------------------------
# Prompt construction
# ---------------------------

def build_system_prompt() -> str:
    return (
        "You are a meticulous financial document VLM assistant. "
        "You read a one-page image of a spreadsheet-like report (pdf export) and extract a SINGLE requested variable. "
        "Rows represent loans and can shift between versions; column headers may act as 'Sections'. "
        "Use the caller-supplied name pointers:\n"
        "- Section (often a column header text or top-of-column label),\n"
        "- Sub-section (vertical positional cue within a column group), and\n"
        "- Line item (horizontal cue within that row or nearby rows).\n\n"
        "You are ALSO given a prior Excel cell hint (e.g., 'M66') from the last version of this report. "
        "Search the 'vicinity' of that cell with a VERTICAL bias first (neighboring rows), and a light HORIZONTAL bias if needed. "
        "If the exact value moved rows this version, it's likely within a few rows above/below. "
        "If columns shifted, look modestly left/right.\n\n"
        "OUTPUT STRICTLY the JSON schema provided in the user instructions. No extra keys, no extra text."
    )

def build_user_prompt(
    variable_name: str,
    pointers: Dict[str, Optional[str]],
    previous_cell_hint: str,
    vertical_window: int = 4,
    horizontal_window: int = 1,
    output_schema_text: str = ""
) -> str:
    """
    pointers: {
        'section': '...',        # often column name
        'sub_section': '...',    # vertical cue
        'line_item': '...'       # horizontal cue
    }
    """
    sec = pointers.get("section") or ""
    sub = pointers.get("sub_section") or ""
    li  = pointers.get("line_item") or ""

    rc = excel_cell_to_rc(previous_cell_hint)
    vicinity_desc = (
        f"Vicinity search (VERTICAL first): within ±{vertical_window} rows of row {rc['row']}; "
        f"if needed HORIZONTAL within ±{horizontal_window} columns of column index {rc['col']}. "
        f"(Previous cell hint: '{previous_cell_hint}')"
    )

    # Concrete instructions to enforce JSON only
    extraction_instructions = f"""
Task:
- Extract the variable: "{variable_name}".

Name Pointers (for localization):
- Section (column-like): "{sec}"
- Sub-section (vertical cue): "{sub}"
- Line item (horizontal cue): "{li}"

Heuristic:
- Prefer vertical scanning near the prior cell location; only adjust columns slightly if needed.
- If multiple candidates are visible, pick the one that best matches ALL the given name pointers.
- Return both the raw text value and a numeric version if applicable (e.g., '12,345.67' -> 12345.67).
- Provide a short evidence snapshot (few words) and your nearest-cell guess (e.g., 'M67' if you can infer it).

STRICT JSON OUTPUT ONLY:
Use exactly this schema (no extra fields, no commentary):
{output_schema_text}
"""

    return f"{vicinity_desc}\n\n{extraction_instructions}".strip()


def json_schema_text_for_model() -> str:
    """
    Text version of the JSON schema for the model to follow.
    """
    return """{
  "variable": string,              // the requested variable name echoed back
  "value_text": string or null,    // the value exactly as seen (raw text)
  "value_numeric": number or null, // parsed numeric if applicable, else null
  "unit": string or null,          // e.g., '%', 'USD', 'bps', or null
  "found": boolean,                // true if you found the value confidently
  "evidence": {
    "section": string or null,         // which section text you used (if any)
    "sub_section": string or null,     // sub-section text used (if any)
    "line_item": string or null,       // line-item text used (if any)
    "cell_text_snapshot": string or null, // a short literal snippet near the value
    "nearest_cell_guess": string or null, // e.g., 'M67' if you can infer
    "coordinates": {
      "x": number or null,
      "y": number or null,
      "w": number or null,
      "h": number or null
    } or null
  } or null,
  "confidence": number or null,    // 0.0-1.0 (your subjective confidence)
  "notes": string or null          // brief notes if needed
}"""


# ---------------------------
# Gemini (via Apigee) call
# ---------------------------

@dataclass
class ApigeeConfig:
    base_url: str               # e.g., "https://<your-apigee-host>/v1/projects/<proj>/locations/<loc>/publishers/google/models/gemini-2.5-pro:generateContent"
    consumer_key: str
    consumer_secret: str
    access_token: str           # already obtained as per user


def build_gemini_request_payload(image_b64: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    """
    Gemini 2.5 (multimodal): content list containing text and image parts.
    Adjust if your Apigee endpoint expects strictly Vertex AI wire format vs. Generative Language API.
    """
    return {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": system_prompt},
                    {"text": user_prompt},
                    {"inline_data": {"mime_type": "image/png", "data": image_b64.split(",")[1]}}
                ]
            }
        ],
        "generation_config": {
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 1024
        },
        "safety_settings": []
    }


def call_gemini(apigee_cfg: ApigeeConfig, payload: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {apigee_cfg.access_token}",
        "Content-Type": "application/json",
        # Apigee gateways often accept either x-api-key or similar headers.
        # Include both key/secret if your proxy expects them.
        "x-api-key": apigee_cfg.consumer_key,
        "x-api-secret": apigee_cfg.consumer_secret,
    }
    resp = requests.post(apigee_cfg.base_url, headers=headers, data=json.dumps(payload), timeout=timeout)
    if resp.status_code >= 400:
        raise RuntimeError(f"Apigee/Gemini request failed: {resp.status_code} {resp.text}")
    return resp.json()


def extract_text_from_gemini_response(resp: Dict[str, Any]) -> str:
    """
    Try the common response layouts. Adjust if your Apigee proxy wraps differently.
    """
    # Generative Language API style:
    try:
        candidates = resp["candidates"]
        parts = candidates[0]["content"]["parts"]
        texts = [p.get("text", "") for p in parts if "text" in p]
        return "\n".join(t for t in texts if t).strip()
    except Exception:
        pass

    # Vertex-style proxy variants:
    try:
        safety = resp.get("promptFeedback")  # ignore
        # Some proxies return { "candidates":[{"content":{"parts":[{"text":"..."}]} }] }
        # Already handled above; otherwise check a direct "text" field:
        if "text" in resp:
            return str(resp["text"]).strip()
    except Exception:
        pass

    # Raw fallback:
    return json.dumps(resp)


# ---------------------------
# JSON Parsing (robust)
# ---------------------------

def recover_json_block(s: str) -> Optional[str]:
    """
    Extract the first plausible {...} block. Handles leading/trailing commentary.
    """
    # Look for the first '{' and last '}' to form a block:
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    block = s[start:end+1]
    return block

def parse_extraction_json(s: str) -> ExtractionResult:
    """
    Attempt strict parse, then recovery if needed.
    """
    # First try direct:
    try:
        return ExtractionResult.model_validate_json(s)
    except Exception:
        pass

    # Try to pull a JSON block:
    block = recover_json_block(s)
    if block:
        # Remove dangerous trailing commas using a light regex (does not fix all issues but helps).
        cleaned = re.sub(r",\s*([}\]])", r"\1", block)
        try:
            return ExtractionResult.model_validate_json(cleaned)
        except ValidationError as ve:
            # Show partial error for debugging
            raise ve
        except Exception:
            # Last attempt: load as dict then validate
            try:
                data = json.loads(cleaned)
                return ExtractionResult.model_validate(data)
            except Exception as e2:
                raise ValueError(f"Failed to parse JSON after cleanup. Raw text:\n{s}\n\nError: {e2}") from e2

    # Nothing worked
    raise ValueError(f"Model did not return JSON. Raw text:\n{s}")


# ---------------------------
# High-level API
# ---------------------------

def extract_variable_from_pdf(
    pdf_path: str,
    variable_name: str,
    pointers: Dict[str, Optional[str]],
    previous_cell_hint: str,
    apigee_base_url: str,
    apigee_consumer_key: str,
    apigee_consumer_secret: str,
    apigee_access_token: str,
    dpi: int = 300,
    vertical_window: int = 4,
    horizontal_window: int = 1,
    max_width: Optional[int] = 2200,
) -> ExtractionResult:
    img_b64 = pdf_to_base64_image(pdf_path, dpi=dpi, max_width=max_width)

    system_prompt = build_system_prompt()
    schema_text = json_schema_text_for_model()
    user_prompt = build_user_prompt(
        variable_name=variable_name,
        pointers=pointers,
        previous_cell_hint=previous_cell_hint,
        vertical_window=vertical_window,
        horizontal_window=horizontal_window,
        output_schema_text=schema_text
    )

    apigee_cfg = ApigeeConfig(
        base_url=apigee_base_url,
        consumer_key=apigee_consumer_key,
        consumer_secret=apigee_consumer_secret,
        access_token=apigee_access_token
    )

    payload = build_gemini_request_payload(img_b64, system_prompt, user_prompt)

    # Basic retry for transient issues / JSON hiccups
    last_text = ""
    for attempt in range(3):
        try:
            resp = call_gemini(apigee_cfg, payload)
            last_text = extract_text_from_gemini_response(resp)
            result = parse_extraction_json(last_text)
            return result
        except Exception as e:
            if attempt == 2:
                raise
            sleep_s = 1.5 * (attempt + 1)
            print(f"[WARN] Attempt {attempt+1} failed ({e}); retrying in {sleep_s:.1f}s...", file=sys.stderr)
            time.sleep(sleep_s)

    # Should not reach here
    raise RuntimeError(f"Failed after retries. Last model text:\n{last_text}")


# ---------------------------
# CLI
# ---------------------------

def _cli():
    import argparse

    parser = argparse.ArgumentParser(description="Extract a Basel risk input from a one-page PDF using Gemini 2.5 Pro via Apigee.")
    parser.add_argument("--pdf", required=True, help="Path to one-page PDF (exported from Excel).")
    parser.add_argument("--variable", required=True, help='Variable name to extract (e.g., "Risk-Weighted Assets").')
    parser.add_argument("--section", default="", help="Section / Column-name pointer (optional).")
    parser.add_argument("--sub_section", default="", help="Sub-section (vertical) pointer (optional).")
    parser.add_argument("--line_item", default="", help="Line-item (horizontal) pointer (optional).")
    parser.add_argument("--prev_cell", required=True, help='Previous version cell hint (e.g., "M66").')
    parser.add_argument("--apigee_base_url", required=True, help="Full Apigee model endpoint URL for generateContent.")
    parser.add_argument("--apigee_consumer_key", required=True)
    parser.add_argument("--apigee_consumer_secret", required=True)
    parser.add_argument("--apigee_access_token", required=True)
    parser.add_argument("--dpi", type=int, default=300, help="PDF render DPI (default: 300).")
    parser.add_argument("--vwin", type=int, default=4, help="Vertical vicinity window in rows (default: 4).")
    parser.add_argument("--hwin", type=int, default=1, help="Horizontal vicinity window in columns (default: 1).")
    parser.add_argument("--max_width", type=int, default=2200, help="Downscale max width in px (default: 2200).")
    args = parser.parse_args()

    pointers = {
        "section": args.section or None,
        "sub_section": args.sub_section or None,
        "line_item": args.line_item or None
    }

    result = extract_variable_from_pdf(
        pdf_path=args.pdf,
        variable_name=args.variable,
        pointers=pointers,
        previous_cell_hint=args.prev_cell,
        apigee_base_url=args.apigee_base_url,
        apigee_consumer_key=args.apigee_consumer_key,
        apigee_consumer_secret=args.apigee_consumer_secret,
        apigee_access_token=args.apigee_access_token,
        dpi=args.dpi,
        vertical_window=args.vwin,
        horizontal_window=args.hwin,
        max_width=args.max_width,
    )

    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    _cli()

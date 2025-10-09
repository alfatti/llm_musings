#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basel Risk Input Extractor (Gemini 2.5 Pro via Apigee)
Now includes request_id and correlation_id for full observability.

Requirements:
    pip install pdf2image pillow pydantic requests
"""

import base64
import io
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests
from pdf2image import convert_from_path
from PIL import Image
from pydantic import BaseModel, Field, ValidationError


# ---------------------------
# Models / Validation
# ---------------------------

class Coordinates(BaseModel):
    x: Optional[float] = None
    y: Optional[float] = None
    w: Optional[float] = None
    h: Optional[float] = None

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
# PDF → Image (Base64)
# ---------------------------

def pdf_to_base64_image(pdf_path: str, dpi: int = 300, max_width: int = 2200) -> str:
    images = convert_from_path(pdf_path, dpi=dpi)
    if not images:
        raise ValueError("No pages found in PDF.")
    if len(images) > 1:
        print("[WARN] Multi-page PDF detected; using first page.", file=sys.stderr)
    img = images[0].convert("RGB")

    if img.width > max_width:
        ratio = max_width / float(img.width)
        img = img.resize((max_width, int(img.height * ratio)), resample=Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


# ---------------------------
# Excel Cell Helpers
# ---------------------------

def excel_col_to_num(col: str) -> int:
    total = 0
    for c in col.upper():
        total = total * 26 + (ord(c) - 64)
    return total

def excel_cell_to_rc(cell: str) -> Dict[str, int]:
    m = re.fullmatch(r"\s*([A-Za-z]+)\s*([0-9]+)\s*", cell)
    if not m:
        raise ValueError(f"Invalid cell reference: {cell}")
    return {"row": int(m.group(2)), "col": excel_col_to_num(m.group(1))}


# ---------------------------
# Prompt Builders
# ---------------------------

def build_system_prompt() -> str:
    return (
        "You are a meticulous financial document VLM assistant. "
        "You analyze a one-page spreadsheet (PDF export) and extract a single requested variable. "
        "Rows represent loans that may shift between versions; column headers act as sections. "
        "Use given name pointers and the previous cell location for vicinity search. "
        "Return only JSON according to the schema provided."
    )

def json_schema_text_for_model() -> str:
    return """{
  "variable": string,
  "value_text": string or null,
  "value_numeric": number or null,
  "unit": string or null,
  "found": boolean,
  "evidence": {
    "section": string or null,
    "sub_section": string or null,
    "line_item": string or null,
    "cell_text_snapshot": string or null,
    "nearest_cell_guess": string or null,
    "coordinates": {"x": number or null,"y": number or null,"w": number or null,"h": number or null} or null
  } or null,
  "confidence": number or null,
  "notes": string or null
}"""

def build_user_prompt(variable_name: str, pointers: Dict[str, Optional[str]],
                      previous_cell_hint: str, vertical_window: int,
                      horizontal_window: int, output_schema_text: str) -> str:
    rc = excel_cell_to_rc(previous_cell_hint)
    return f"""
Variable: {variable_name}
Section: {pointers.get("section")}
Sub-section: {pointers.get("sub_section")}
Line item: {pointers.get("line_item")}

Heuristic: Search within ±{vertical_window} rows (vertical bias) and ±{horizontal_window} columns (horizontal bias)
around previous cell {previous_cell_hint} (R{rc['row']},C{rc['col']}).

Output JSON only:
{output_schema_text}
""".strip()


# ---------------------------
# Apigee Config and Call
# ---------------------------

@dataclass
class ApigeeConfig:
    base_url: str
    consumer_key: str
    consumer_secret: str
    access_token: str


def build_gemini_payload(image_b64: str, system_prompt: str, user_prompt: str,
                         request_id: str, correlation_id: str) -> Dict[str, Any]:
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
            "max_output_tokens": 1024,
            "metadata": {
                "request_id": request_id,
                "correlation_id": correlation_id
            }
        },
        "safety_settings": []
    }


def call_gemini(apigee_cfg: ApigeeConfig, payload: Dict[str, Any],
                request_id: str, correlation_id: str, timeout: int = 60) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {apigee_cfg.access_token}",
        "Content-Type": "application/json",
        "x-api-key": apigee_cfg.consumer_key,
        "x-api-secret": apigee_cfg.consumer_secret,
        "x-request-id": request_id,
        "x-correlation-id": correlation_id,
    }
    r = requests.post(apigee_cfg.base_url, headers=headers, json=payload, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"API error {r.status_code}: {r.text}")
    return r.json()


# ---------------------------
# JSON Utilities
# ---------------------------

def extract_text_from_response(resp: Dict[str, Any]) -> str:
    try:
        candidates = resp["candidates"]
        return "\n".join(p["text"] for p in candidates[0]["content"]["parts"] if "text" in p).strip()
    except Exception:
        return json.dumps(resp)

def recover_json_block(s: str) -> Optional[str]:
    start, end = s.find("{"), s.rfind("}")
    return s[start:end+1] if start >= 0 and end > start else None

def parse_extraction_json(s: str) -> ExtractionResult:
    try:
        return ExtractionResult.model_validate_json(s)
    except Exception:
        block = recover_json_block(s)
        if not block:
            raise ValueError(f"Cannot parse JSON: {s}")
        cleaned = re.sub(r",\s*([}\]])", r"\1", block)
        return ExtractionResult.model_validate_json(cleaned)


# ---------------------------
# Core Function
# ---------------------------

def extract_variable_from_pdf(pdf_path: str, variable_name: str,
    pointers: Dict[str, Optional[str]], previous_cell_hint: str,
    apigee_cfg: ApigeeConfig, dpi: int = 300,
    vertical_window: int = 4, horizontal_window: int = 1,
    correlation_id: Optional[str] = None) -> ExtractionResult:

    # UUIDs
    request_id = str(uuid.uuid4())
    correlation_id = correlation_id or str(uuid.uuid4())
    print(f"[INFO] request_id={request_id}, correlation_id={correlation_id}")

    image_b64 = pdf_to_base64_image(pdf_path, dpi=dpi)
    sys_prompt = build_system_prompt()
    schema = json_schema_text_for_model()
    user_prompt = build_user_prompt(variable_name, pointers, previous_cell_hint,
                                    vertical_window, horizontal_window, schema)

    payload = build_gemini_payload(image_b64, sys_prompt, user_prompt,
                                   request_id, correlation_id)

    raw_resp = call_gemini(apigee_cfg, payload, request_id, correlation_id)
    raw_text = extract_text_from_response(raw_resp)
    return parse_extraction_json(raw_text)


# ---------------------------
# CLI
# ---------------------------

def _cli():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--variable", required=True)
    parser.add_argument("--section", default="")
    parser.add_argument("--sub_section", default="")
    parser.add_argument("--line_item", default="")
    parser.add_argument("--prev_cell", required=True)
    parser.add_argument("--apigee_base_url", required=True)
    parser.add_argument("--apigee_consumer_key", required=True)
    parser.add_argument("--apigee_consumer_secret", required=True)
    parser.add_argument("--apigee_access_token", required=True)
    parser.add_argument("--corr_id", default=None)
    args = parser.parse_args()

    apigee_cfg = ApigeeConfig(
        base_url=args.apigee_base_url,
        consumer_key=args.apigee_consumer_key,
        consumer_secret=args.apigee_consumer_secret,
        access_token=args.apigee_access_token
    )

    pointers = {
        "section": args.section or None,
        "sub_section": args.sub_section or None,
        "line_item": args.line_item or None
    }

    res = extract_variable_from_pdf(
        pdf_path=args.pdf,
        variable_name=args.variable,
        pointers=pointers,
        previous_cell_hint=args.prev_cell,
        apigee_cfg=apigee_cfg,
        correlation_id=args.corr_id
    )
    print(json.dumps(res.model_dump(), indent=2))


if __name__ == "__main__":
    _cli()

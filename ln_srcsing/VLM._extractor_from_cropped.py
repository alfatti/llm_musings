#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basel Risk Input Extractor — OpenAI-style Apigee Proxy (Gemini 2.5 Pro)

- Calls:   {BASE_URL}/v1/chat/completions
- Model:   passed as "model": "gemini-2.5-pro" (or your configured name)
- Headers: x-wf-api-key, x-wf-usecase-id, x-wf-client-id, x-request-id, x-correlation-id, x-wf-request-date
- Auth:    Authorization: Bearer <APIGEE_ACCESS_TOKEN> (fetched via Basic auth if not provided)
- Input:   one-page PDF -> PNG (base64 data URL) + schema-driven prompt
- Output:  strict JSON with (value_text/value_numeric/unit/found/evidence/confidence/notes)

Requirements:
    pip install pdf2image pillow pydantic requests
Plus: Poppler must be on PATH for pdf2image.
"""

import base64
import io
import json
import re
import sys
import time
import uuid
import typing as T
from dataclasses import dataclass
from datetime import datetime, timezone

import requests
from pdf2image import convert_from_path
from PIL import Image
from pydantic import BaseModel, Field, ValidationError


# =========================
# Models / Validation
# =========================

class Coordinates(BaseModel):
    x: T.Optional[float] = None
    y: T.Optional[float] = None
    w: T.Optional[float] = None
    h: T.Optional[float] = None

class Evidence(BaseModel):
    section: T.Optional[str] = None
    sub_section: T.Optional[str] = None
    line_item: T.Optional[str] = None
    cell_text_snapshot: T.Optional[str] = None
    nearest_cell_guess: T.Optional[str] = None
    coordinates: T.Optional[Coordinates] = None

class ExtractionResult(BaseModel):
    variable: str
    value_text: T.Optional[str] = None
    value_numeric: T.Optional[float] = None
    unit: T.Optional[str] = None
    found: bool
    evidence: T.Optional[Evidence] = None
    confidence: T.Optional[float] = Field(default=None, ge=0.0, le=1.0)
    notes: T.Optional[str] = None


# =========================
# Utilities
# =========================

def rfc3339_now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def pdf_to_base64_image(pdf_path: str, dpi: int = 300, max_width: int = 2200) -> str:
    pages = convert_from_path(pdf_path, dpi=dpi)
    if not pages:
        raise ValueError("No pages found in PDF.")
    if len(pages) > 1:
        print("[WARN] Multi-page PDF detected; using first page.", file=sys.stderr)
    img = pages[0].convert("RGB")
    if img.width > max_width:
        ratio = max_width / float(img.width)
        img = img.resize((max_width, int(img.height * ratio)), resample=Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

def excel_col_to_num(col: str) -> int:
    total = 0
    for c in col.upper():
        total = total * 26 + (ord(c) - 64)
    return total

def excel_cell_to_rc(cell: str) -> dict:
    m = re.fullmatch(r"\s*([A-Za-z]+)\s*([0-9]+)\s*", cell)
    if not m:
        raise ValueError(f"Invalid cell reference: {cell}")
    return {"row": int(m.group(2)), "col": excel_col_to_num(m.group(1))}


# =========================
# Prompt builders
# =========================

def build_system_prompt() -> str:
    return (
        "You are a meticulous financial document VLM assistant. "
        "Analyze a one-page spreadsheet (PDF export) and extract a single requested variable. "
        "Rows represent loans and may shift between versions; column headers act as sections. "
        "Use the provided name pointers (Section, Sub-section, Line item) and the prior Excel cell hint. "
        "Search vertically near the prior row first, then minimally across columns. "
        "Return ONLY JSON that matches the schema exactly."
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

def build_user_prompt(variable_name: str, pointers: dict,
                      previous_cell_hint: str, vwin: int, hwin: int,
                      schema_text: str) -> str:
    rc = excel_cell_to_rc(previous_cell_hint)
    return f"""
Variable: {variable_name}
Section: {pointers.get("section")}
Sub-section: {pointers.get("sub_section")}
Line item: {pointers.get("line_item")}

Heuristic: Search within ±{vwin} rows (vertical bias) and ±{hwin} columns (horizontal bias)
around previous cell {previous_cell_hint} (R{rc['row']},C{rc['col']}).

Output JSON only:
{schema_text}
""".strip()


# =========================
# Apigee / OpenAI-style proxy
# =========================

@dataclass
class ProxyCfg:
    # OpenAI-style base URL, e.g., "https://apigee.company.com/v1"
    base_url: str
    # Model name, e.g., "gemini-2.5-pro"
    model: str
    # WF app identity headers
    api_key: str         # x-wf-api-key
    usecase_id: str      # x-wf-usecase-id
    app_id: str          # x-wf-client-id
    # Auth: either give access_token, or supply login_url + consumer_key + consumer_secret
    access_token: T.Optional[str] = None
    login_url: T.Optional[str] = None
    consumer_key: T.Optional[str] = None
    consumer_secret: T.Optional[str] = None

def ensure_access_token(cfg: ProxyCfg) -> str:
    if cfg.access_token:
        return cfg.access_token
    if not (cfg.login_url and cfg.consumer_key and cfg.consumer_secret):
        raise ValueError("Either provide access_token OR (login_url, consumer_key, consumer_secret).")
    import base64 as b64
    pair = f"{cfg.consumer_key}:{cfg.consumer_secret}".encode("utf-8")
    basic = b64.b64encode(pair).decode("utf-8")
    resp = requests.post(
        cfg.login_url,
        headers={"Authorization": f"Basic {basic}"},
        data={},  # add form fields if your token endpoint requires them
        timeout=30
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"Apigee login failed {resp.status_code}: {resp.text}")
    data = resp.json()
    token = data.get("access_token") or data.get("acces_token")  # guard spelling
    if not token:
        raise RuntimeError(f"Apigee login response missing access token: {data}")
    cfg.access_token = token
    return token

def build_headers(cfg: ProxyCfg, request_id: str, correlation_id: str, request_date: str) -> dict:
    token = ensure_access_token(cfg)
    return {
        "x-wf-api-key": cfg.api_key,
        "x-wf-usecase-id": cfg.usecase_id,
        "x-wf-client-id": cfg.app_id,
        "x-request-id": request_id,           # note: hyphen, not underscore
        "x-correlation-id": correlation_id,
        "x-wf-request-date": request_date,    # RFC3339 UTC
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }

def build_openai_style_payload(model: str, system_prompt: str,
                               user_prompt: str, image_data_url: str) -> dict:
    """
    OpenAI-style /v1/chat/completions payload.
    Many proxies accept 'input_image' with image_url as a data URL.
    """
    return {
        "model": model,
        "temperature": 0.2,
        "top_p": 0.95,
        "max_tokens": 1024,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "input_image", "image_url": image_data_url}
                ]
            }
        ]
    }

def call_openai_style(cfg: ProxyCfg, headers: dict, payload: dict, timeout: int = 60) -> dict:
    url = cfg.base_url.rstrip("/") + "/v1/chat/completions"
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"Model call failed {r.status_code}: {r.text}")
    return r.json()

def extract_text_from_openai_response(resp: dict) -> str:
    """
    Try to pull the text from OpenAI-style choices.
    Some proxies wrap differently; this handles the common case:
      { choices: [ { message: { content: "..." } } ] }
    """
    try:
        return resp["choices"][0]["message"]["content"]
    except Exception:
        # Fallback (rare proxy variants)
        return json.dumps(resp)


# =========================
# JSON parsing (robust)
# =========================

def recover_json_block(s: str) -> T.Optional[str]:
    start, end = s.find("{"), s.rfind("}")
    return s[start:end+1] if (start >= 0 and end > start) else None

def parse_extraction_json(s: str) -> ExtractionResult:
    # attempt direct pydantic parse
    try:
        return ExtractionResult.model_validate_json(s)
    except Exception:
        block = recover_json_block(s)
        if not block:
            raise ValueError(f"Cannot parse JSON: {s}")
        # soften trailing commas
        cleaned = re.sub(r",\s*([}\]])", r"\1", block)
        try:
            return ExtractionResult.model_validate_json(cleaned)
        except ValidationError as ve:
            # last resort: json.loads then validate
            data = json.loads(cleaned)
            return ExtractionResult.model_validate(data)


# =========================
# High-level API
# =========================

def extract_variable_from_pdf(
    pdf_path: str,
    variable_name: str,
    pointers: dict,
    previous_cell_hint: str,
    cfg: ProxyCfg,
    dpi: int = 300,
    vwin: int = 4,
    hwin: int = 1,
) -> ExtractionResult:

    # Match your sample: same UUID for request & correlation
    req_id = corr_id = str(uuid.uuid4())
    req_date = rfc3339_now_utc()
    print(f"[INFO] request_id={req_id} correlation_id={corr_id} x-wf-request-date={req_date}")

    img_b64_url = pdf_to_base64_image(pdf_path, dpi=dpi)
    sys_prompt = build_system_prompt()
    schema = json_schema_text_for_model()
    user_prompt = build_user_prompt(variable_name, pointers, previous_cell_hint, vwin, hwin, schema)

    headers = build_headers(cfg, req_id, corr_id, req_date)
    payload = build_openai_style_payload(cfg.model, sys_prompt, user_prompt, img_b64_url)

    # retry (keep same IDs for traceability; refresh request date)
    last_text = ""
    for attempt in range(3):
        try:
            resp = call_openai_style(cfg, headers, payload)
            last_text = extract_text_from_openai_response(resp)
            return parse_extraction_json(last_text)
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(1.2 * (attempt + 1))
            headers["x-wf-request-date"] = rfc3339_now_utc()
            print(f"[WARN] attempt {attempt+1} failed: {e}; retrying...", file=sys.stderr)

    raise RuntimeError(f"Failed after retries. Last model text:\n{last_text}")


# =========================
# CLI
# =========================

def _cli():
    import argparse
    p = argparse.ArgumentParser(description="Basel variable extractor via OpenAI-style Apigee proxy (Gemini 2.5 Pro).")
    p.add_argument("--pdf", required=True, help="Path to one-page PDF (export from Excel).")
    p.add_argument("--variable", required=True, help='Variable to extract, e.g., "Risk-Weighted Assets".')
    p.add_argument("--section", default="", help="Section / column-name pointer (optional).")
    p.add_argument("--sub_section", default="", help="Sub-section (vertical cue) (optional).")
    p.add_argument("--line_item", default="", help="Line-item (horizontal cue) (optional).")
    p.add_argument("--prev_cell", required=True, help='Previous version cell hint, e.g., "M66".')

    # Proxy config
    p.add_argument("--base_url", required=True, help="OpenAI-style Apigee base URL, e.g., https://apigee.company.com/v1")
    p.add_argument("--model", required=True, help='Model name, e.g., "gemini-2.5-pro"')

    # WF headers
    p.add_argument("--api_key", required=True, help="x-wf-api-key")
    p.add_argument("--usecase_id", required=True, help="x-wf-usecase-id")
    p.add_argument("--app_id", required=True, help="x-wf-client-id")

    # Auth: either --access_token OR ( --login_url + --consumer_key + --consumer_secret )
    p.add_argument("--access_token", default=None)
    p.add_argument("--login_url", default=None)
    p.add_argument("--consumer_key", default=None)
    p.add_argument("--consumer_secret", default=None)

    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--vwin", type=int, default=4)
    p.add_argument("--hwin", type=int, default=1)
    args = p.parse_args()

    cfg = ProxyCfg(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        usecase_id=args.usecase_id,
        app_id=args.app_id,
        access_token=args.access_token,
        login_url=args.login_url,
        consumer_key=args.consumer_key,
        consumer_secret=args.consumer_secret
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
        cfg=cfg,
        dpi=args.dpi,
        vwin=args.vwin,
        hwin=args.hwin
    )

    print(res.model_dump_json(indent=2))


if __name__ == "__main__":
    _cli()
#================================================================


# 1) install
pip install pdf2image pillow pydantic requests

# 2) run (token fetched automatically via Basic auth)
python basel_variable_extractor_openai_proxy.py \
  --pdf /path/to/one_page.pdf \
  --variable "Risk-Weighted Assets" \
  --section "RWA" \
  --sub_section "Basel III Standardized" \
  --line_item "Total" \
  --prev_cell "M66" \
  --base_url "https://apigee.company.com/v1" \
  --model "gemini-2.5-pro" \
  --api_key "$API_KEY" \
  --usecase_id "$USECASE_ID" \
  --app_id "$APP_ID" \
  --login_url "$APIGEE_LOGIN_URL" \
  --consumer_key "$APIGEE_CONSUMER_KEY" \
  --consumer_secret "$APIGEE_CONSUMER_SECRET"

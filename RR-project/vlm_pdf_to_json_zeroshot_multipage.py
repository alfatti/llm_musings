#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Zero-shot VLM pipeline for multi-page PDF → JSON extraction with Gemini (Vertex AI, REST).
- Shred PDF to per-page PNG bytes (in-memory)
- Parallel LLM calls (ThreadPoolExecutor)
- Robust JSON parsing + light repair + retries
- Consolidated results

Prereqs:
  pip install pdf2image pillow requests google-auth
  # Also install Poppler for pdf2image (macOS: brew install poppler; Ubuntu: apt-get install poppler-utils)

Auth:
  - Easiest: export ACCESS_TOKEN from gcloud
      gcloud auth application-default login
      export VERTEX_ACCESS_TOKEN="$(gcloud auth print-access-token)"
  - Or let google-auth fetch ADC automatically (service account / workload identity).

Run:
  python vlm_pdf_to_json.py input.pdf --project YOUR_GCP_PROJECT --location us-central1 --model gemini-1.5-pro
"""

from __future__ import annotations
import os, io, re, json, base64, time, math, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
from pdf2image import convert_from_path, convert_from_bytes
from PIL import Image

try:
    # Prefer ADC if available
    from google.auth.transport.requests import Request as GAuthRequest
    from google.oauth2 import service_account
    import google.auth
except Exception:
    google = None  # type: ignore

# -----------------------------
# Configuration
# -----------------------------
DEFAULT_MODEL = "gemini-1.5-pro"
DEFAULT_LOCATION = "us-central1"
MAX_WORKERS = min(8, os.cpu_count() or 4)
PER_PAGE_MAX_RETRIES = 2
RATE_LIMIT_SLEEP_BASE = 1.5  # exponential backoff base seconds
TIMEOUT_SECS = 120

ZERO_SHOT_PROMPT = (
    "Extract a clean JSON array describing all table rows on this rent-roll page. "
    "Return ONLY JSON (no text) and ensure valid JSON syntax.\n\n"
    "Target schema (use available fields; if a field is missing, omit it rather than null):\n"
    "[{\n"
    '  "Property": str,\n'
    '  "Unit": str,\n'
    '  "Tenant": str,\n'
    '  "Status": str,\n'
    '  "SQFT": number,\n'
    '  "Bedrooms": number,\n'
    '  "Bathrooms": number,\n'
    '  "LeaseFrom": str,  // ISO or as seen\n'
    '  "LeaseTo": str,\n'
    '  "ChargeTypes": { "Rent": number, "...": number },\n'
    '  "TotalScheduled": number\n'
    "}]\n"
    "Do not include prose or markdown fences—JSON only."
)

# -----------------------------
# Utilities
# -----------------------------

def get_access_token() -> Optional[str]:
    """
    Obtain an OAuth2 access token for Vertex AI.
    Priority:
      1) env VERTEX_ACCESS_TOKEN
      2) Application Default Credentials via google-auth (if installed)
    """
    token = os.environ.get("VERTEX_ACCESS_TOKEN")
    if token:
        return token.strip()

    if 'google' in globals() and google is not None:
        try:
            creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            if not creds.valid:
                creds.refresh(GAuthRequest())
            return creds.token
        except Exception:
            pass
    return None


def png_bytes_from_pil(img: Image.Image, dpi: int = 200) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def shred_pdf_to_png_bytes(
    pdf_path: Optional[str] = None,
    pdf_bytes: Optional[bytes] = None,
    dpi: int = 200
) -> List[bytes]:
    """
    Convert multi-page PDF to a list of PNG bytes (one per page).
    Uses pdf2image; requires Poppler.
    """
    assert pdf_path or pdf_bytes, "Provide either pdf_path or pdf_bytes"
    if pdf_path:
        pages = convert_from_path(pdf_path, dpi=dpi)
    else:
        pages = convert_from_bytes(pdf_bytes, dpi=dpi)
    return [png_bytes_from_pil(p) for p in pages]


def b64_png(png_bytes: bytes) -> str:
    return base64.b64encode(png_bytes).decode("utf-8")


def vertex_gemini_url(project: str, location: str, model: str) -> str:
    # Generative Language (Vertex) v1beta REST endpoint
    return (
        f"https://{location}-aiplatform.googleapis.com/v1beta/"
        f"projects/{project}/locations/{location}/publishers/google/models/{model}:generateContent"
    )


def build_request_body(image_png_b64: str, prompt_text: str) -> Dict[str, Any]:
    """
    Minimal Vertex request body with one image and one text part (zero-shot).
    """
    return {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": image_png_b64
                        }
                    },
                    {"text": prompt_text}
                ]
            }
        ],
        # You can tweak generationConfig if desired
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 2048,
        }
    }


def parse_json_strict_maybe(text: str) -> Any:
    """
    Try to parse JSON strictly. If it fails, try to:
      - strip code fences
      - find the first {...} or [...] balanced block
    """
    # Quick accept
    try:
        return json.loads(text)
    except Exception:
        pass

    # Strip ```...``` fences if present
    text2 = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        return json.loads(text2)
    except Exception:
        pass

    # Try to find the first top-level JSON array/object via a heuristic
    candidate = extract_first_json_block(text2)
    if candidate:
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # One last try on raw text (LLM sometimes returns pure JSON with leading/trailing spaces)
    return json.loads(text)  # will raise


def extract_first_json_block(s: str) -> Optional[str]:
    """
    Heuristic: find the first balanced {...} or [...] block at top level.
    """
    for opener, closer in [("{", "}"), ("[", "]")]:
        start_idx = s.find(opener)
        if start_idx == -1:
            continue
        depth = 0
        for i in range(start_idx, len(s)):
            if s[i] == opener:
                depth += 1
            elif s[i] == closer:
                depth -= 1
                if depth == 0:
                    return s[start_idx:i+1]
    return None


@dataclass
class PageResult:
    page_index: int   # 0-based
    ok: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    raw_text: Optional[str] = None


def call_gemini_vertex_once(
    url: str,
    access_token: str,
    request_body: Dict[str, Any],
    timeout: int = TIMEOUT_SECS
) -> str:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json; charset=utf-8",
    }
    resp = requests.post(url, headers=headers, json=request_body, timeout=timeout)
    if resp.status_code == 429:
        raise RuntimeError("Rate limited (429).")
    if resp.status_code >= 400:
        raise RuntimeError(f"Vertex error {resp.status_code}: {resp.text[:500]}")

    j = resp.json()
    # Standard Vertex response path
    try:
        return j["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        # Sometimes responses differ; surface the JSON for debugging
        raise RuntimeError(f"Unexpected response shape: {json.dumps(j)[:800]}")


def extract_page_json_with_retries(
    page_idx: int,
    image_b64: str,
    prompt_text: str,
    url: str,
    access_token: str,
    max_retries: int = PER_PAGE_MAX_RETRIES,
) -> PageResult:
    delay = RATE_LIMIT_SLEEP_BASE
    for attempt in range(max_retries + 1):
        try:
            body = build_request_body(image_b64, prompt_text)
            text = call_gemini_vertex_once(url, access_token, body)
            parsed = parse_json_strict_maybe(text)
            # Accept either a list or an object; up to you. Here we accept both.
            if not isinstance(parsed, (list, dict)):
                raise ValueError("Model returned non-JSON-collection type.")
            return PageResult(page_index=page_idx, ok=True, data=parsed, raw_text=text)
        except Exception as e:
            last_err = str(e)
            # Exponential backoff on any error (esp. 429 or transient)
            if attempt < max_retries:
                time.sleep(delay)
                delay *= 2.0
            else:
                return PageResult(page_index=page_idx, ok=False, error=last_err, raw_text=None)


def run_pdf_extraction(
    pdf_path: Optional[str],
    pdf_bytes: Optional[bytes],
    project: str,
    location: str = DEFAULT_LOCATION,
    model: str = DEFAULT_MODEL,
    prompt_text: str = ZERO_SHOT_PROMPT,
    max_workers: int = MAX_WORKERS,
) -> List[PageResult]:
    # 1) Shred PDF
    png_pages = shred_pdf_to_png_bytes(pdf_path=pdf_path, pdf_bytes=pdf_bytes, dpi=200)
    if not png_pages:
        raise RuntimeError("No pages found in PDF.")

    # 2) Auth + URL
    access_token = get_access_token()
    if not access_token:
        raise RuntimeError(
            "No Vertex access token. Export VERTEX_ACCESS_TOKEN or configure ADC (gcloud / service account)."
        )
    url = vertex_gemini_url(project, location, model)

    # 3) Parallel page processing
    results: List[PageResult] = [None] * len(png_pages)  # type: ignore
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = []
        for i, png in enumerate(png_pages):
            b64img = b64_png(png)
            futures.append(
                ex.submit(
                    extract_page_json_with_retries,
                    i, b64img, prompt_text, url, access_token, PER_PAGE_MAX_RETRIES
                )
            )
        for fut in as_completed(futures):
            r = fut.result()
            results[r.page_index] = r

    return results


def flatten_results(results: List[PageResult]) -> List[Dict[str, Any]]:
    """
    Flatten per-page JSON into a single list of records.
    If a page returns a dict, we wrap it; if a list, we extend.
    Adds `__page` metadata.
    """
    out: List[Dict[str, Any]] = []
    for r in results:
        if not r.ok or r.data is None:
            continue
        if isinstance(r.data, dict):
            item = dict(r.data)
            item["__page"] = r.page_index + 1
            out.append(item)
        elif isinstance(r.data, list):
            for row in r.data:
                if isinstance(row, dict):
                    row2 = dict(row)
                    row2["__page"] = r.page_index + 1
                    out.append(row2)
                else:
                    out.append({"__page": r.page_index + 1, "__value": row})
        else:
            out.append({"__page": r.page_index + 1, "__value": r.data})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", help="Path to input PDF")
    ap.add_argument("--project", required=True, help="GCP project ID")
    ap.add_argument("--location", default=DEFAULT_LOCATION, help="Vertex location (e.g., us-central1)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model (e.g., gemini-1.5-pro)")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS, help="Max parallel workers")
    ap.add_argument("--out_json", default="extracted.json", help="Where to save flattened JSON")
    args = ap.parse_args()

    results = run_pdf_extraction(
        pdf_path=args.pdf,
        pdf_bytes=None,
        project=args.project,
        location=args.location,
        model=args.model,
        max_workers=args.workers,
    )

    # Log errors (if any)
    errors = [r for r in results if not r.ok]
    if errors:
        print(f"[WARN] {len(errors)} pages failed:")
        for e in errors:
            print(f"  - page {e.page_index+1}: {e.error}")

    flat = flatten_results(results)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(flat, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved {len(flat)} records to {args.out_json}")


if __name__ == "__main__":
    main()


# Notes & knobs you can tweak
# Zero-shot: ZERO_SHOT_PROMPT is intentionally lean; add/remove fields freely without changing code.
# Parallelism: change MAX_WORKERS or --workers based on your quota; for very large PDFs, keep it moderate (4–8).
# Retries & Backoff: PER_PAGE_MAX_RETRIES and RATE_LIMIT_SLEEP_BASE tame transient 429/5xxs.
# Auth: quickest path is export VERTEX_ACCESS_TOKEN="$(gcloud auth print-access-token)". The script will also try ADC if google-auth is installed and ADC is configured.




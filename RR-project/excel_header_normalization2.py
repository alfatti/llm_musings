#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLM-driven header clipping for Excel rent rolls.
- Input: a PDF (single page) or image of the printed sheet that includes Excel row numbers and column letters.
- Output: executes model-generated pandas code to create df_normalized.
Usage:
  python vlm_header_clip.py \
      --image_or_pdf "/path/to/printed_sheet.pdf" \
      --rr_path "/path/to/original.xlsx" \
      --sheet_name "RentRoll" \
      --provider "gemini" \
      --mode "vertex" \
      --model "gemini-2.0-pro-exp-02-05" \
      --project "YOUR_GCP_PROJECT" \
      --location "us-central1"
Notes:
- Set one of:
  - GEMINI_API_KEY (for mode=api)
  - Or use Vertex (mode=vertex) with ADC (gcloud auth application-default login) or service account.
"""

import os
import re
import io
import sys
import json
import base64
import argparse
from typing import Optional, Tuple

# Optional imports for PDF->image conversion
# Supports either pdf2image(poppler) or PyMuPDF (fitz)
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}

def load_image_bytes(image_or_pdf: str) -> bytes:
    p = image_or_pdf.lower()
    if any(p.endswith(ext) for ext in IMG_EXTS):
        with open(image_or_pdf, "rb") as f:
            return f.read()
    # Try PDF first page to PNG via PyMuPDF if available, else pdf2image
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(image_or_pdf)
        page = doc.load_page(0)
        pix = page.get_pixmap()  # default DPI; adjust if needed
        img_bytes = pix.tobytes("png")
        return img_bytes
    except Exception:
        pass
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(image_or_pdf, first_page=1, last_page=1)
        if not images:
            raise RuntimeError("pdf2image returned no pages.")
        buf = io.BytesIO()
        images[0].save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        raise RuntimeError(f"Failed to render PDF to image: {e}")

def build_prompt() -> str:
    """
    Prompt requires the model to read the printed sheet that includes
    Excel row numbers and column letters, and then OUTPUT ONLY a python code block.
    The code must:
      - Use EXCEL-FIRST-DATA ROW (by Excel visible row index) determined from the image.
      - Optionally use EXCEL_HEADER_ROWS (list) and EXCEL_LAST_DATA_ROW if seen.
      - Read RR_PATH (+ SHEET_NAME if given), slice rows properly, promote headers, clean.
      - Produce df_normalized (a vanilla table) and NOT print extraneous text.
    """
    return (
        "You are given an IMAGE of a printed Excel sheet (PDF export) where "
        "Excel row numbers (1,2,3,...) and column letters (A,B,C,...) are visible.\n\n"
        "Your task: infer exactly where the DATA BODY begins using those visible Excel indices, "
        "and generate EXECUTABLE pandas code that:\n"
        "1) Uses the Excel-visible **first data row number** (e.g., 14 if the first true data row is Excel row 14).\n"
        "2) Optionally uses the Excel-visible **last data row** if a clear footer is visible (e.g., totals, footnotes).\n"
        "3) Promotes the true column headers to the DataFrame columns. If headers are split across multiple printed "
        "rows (e.g., two-line headers), combine them into one using ' - ' between pieces.\n"
        "4) Trims whitespace, drops empty columns/rows, de-dups column names, and returns a clean rectangular table named **df_normalized**.\n\n"
        "IMPORTANT CONSTRAINTS:\n"
        "- Assume you CANNOT read the image at runtime; your job is to look at the image NOW and hardcode the relevant row indices.\n"
        "- The resulting code must read the underlying Excel at RR_PATH (and SHEET_NAME if provided) and slice using pandas iloc.\n"
        "- If there are multiple header lines above the data body, construct the final header by combining those rows.\n"
        "- If you see a footer after the data (e.g., totals), drop those rows.\n"
        "- NEVER guess paths. Use RR_PATH and optional SHEET_NAME variables that are injected at runtime.\n"
        "- Output ONLY a single python code block fenced as ```python ... ``` with NO extra commentary.\n\n"
        "Return code requirements:\n"
        "- Define integers: EXCEL_FIRST_DATA_ROW, and (if applicable) EXCEL_LAST_DATA_ROW.\n"
        "- Optionally define EXCEL_HEADER_ROWS as a list of row numbers (top-to-bottom) that together form the header.\n"
        "- Read the first sheet if SHEET_NAME is None; otherwise target SHEET_NAME.\n"
        "- The code must end with a df_normalized variable in memory.\n"
        "- Avoid non-determinism and avoid printing anything except optionally df_normalized.head().\n"
    )

def call_gemini_vertex(image_bytes: bytes, prompt: str, model: str, project: str, location: str) -> str:
    # Vertex AI (Generative) path
    try:
        from vertexai.generative_models import GenerativeModel, Part, SafetySetting
        import vertexai
    except Exception as e:
        raise RuntimeError("vertexai package not available. Install google-cloud-aiplatform>=1.64 and vertexai extras.") from e

    vertexai.init(project=project, location=location)
    gen = GenerativeModel(model)
    img_part = Part.from_data(data=image_bytes, mime_type="image/png")
    resp = gen.generate_content(
        [prompt, img_part],
        safety_settings=[
            SafetySetting(category=SafetySetting.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=SafetySetting.HarmBlockThreshold.BLOCK_NONE),
        ],
        generation_config={"temperature": 0.2, "max_output_tokens": 2048},
    )
    return resp.text or ""

def call_gemini_api(image_bytes: bytes, prompt: str, model: str) -> str:
    # Gemini API via google-generativeai
    try:
        import google.generativeai as genai
    except Exception as e:
        raise RuntimeError("google-generativeai not installed. pip install google-generativeai") from e

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY for mode=api")

    genai.configure(api_key=api_key)
    model_obj = genai.GenerativeModel(model)
    # Inline image as a dict
    image_part = {
        "mime_type": "image/png",
        "data": image_bytes,
    }
    resp = model_obj.generate_content(
        [prompt, image_part],
        generation_config={"temperature": 0.2, "max_output_tokens": 2048}
    )
    return getattr(resp, "text", "") or ""

CODE_BLOCK_RE = re.compile(r"```python\s*(.*?)```", re.DOTALL | re.IGNORECASE)

def extract_python_block(text: str) -> str:
    m = CODE_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()
    # fallback: try any fenced block
    m2 = re.search(r"```[\w]*\s*(.*?)```", text, re.DOTALL)
    if m2:
        return m2.group(1).strip()
    # last resort: if it looks like raw code, return as-is
    return text.strip()

def safe_exec(code: str, rr_path: str, sheet_name: Optional[str]) -> dict:
    """
    Execute model code with restricted globals.
    We inject RR_PATH and SHEET_NAME.
    Expect df_normalized in locals at the end.
    """
    # VERY minimal sandbox (still executes arbitrary code—use only in controlled environments)
    safe_globals = {
        "__builtins__": {
            "True": True, "False": False, "None": None,
            "len": len, "range": range, "min": min, "max": max, "sum": sum,
            "print": print, "enumerate": enumerate, "list": list, "dict": dict, "set": set,
            "any": any, "all": all, "zip": zip, "abs": abs, "round": round,
        }
    }
    safe_locals = {}
    # Inject variables the model must use
    safe_locals["RR_PATH"] = rr_path
    safe_locals["SHEET_NAME"] = sheet_name

    # Provide pandas, numpy by default so model code can import/use quickly
    import pandas as pd  # noqa
    import numpy as np   # noqa
    safe_locals["pd"] = pd
    safe_locals["np"] = np

    exec(code, safe_globals, safe_locals)
    return safe_locals

def heuristic_fallback(rr_path: str, sheet_name: Optional[str]):
    """
    Very simple fallback: guess header row as the first row with the largest count of non-null string-like cells.
    Promotes that row to header and cleans basic stuff.
    """
    import pandas as pd
    df = pd.read_excel(rr_path, sheet_name=sheet_name or 0, header=None, dtype=str)
    # Find candidate header row
    best_idx = 0
    best_score = -1
    for i in range(min(len(df), 40)):  # scan first 40 rows
        row = df.iloc[i].astype(str)
        score = (row.notna() & row.str.strip().ne("")).sum()
        if score > best_score:
            best_score = score
            best_idx = i
    body = df.iloc[best_idx+1:].copy()
    header_raw = df.iloc[best_idx].fillna("").astype(str).str.strip().tolist()
    body.columns = [c.strip() for c in header_raw]
    # Clean
    body = body.dropna(how="all", axis=0)
    body = body.dropna(how="all", axis=1)
    body.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in body.columns]
    # Dedup columns
    seen = {}
    new_cols = []
    for c in body.columns:
        k = c
        if k in seen:
            seen[k] += 1
            k = f"{k}_{seen[c]}"
        else:
            seen[k] = 0
        new_cols.append(k)
    body.columns = new_cols
    return body

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_or_pdf", required=True, help="Path to printed Excel image or 1-page PDF (with visible row/col indices).")
    ap.add_argument("--rr_path", required=True, help="Path to original Excel/CSV file to read with pandas.")
    ap.add_argument("--sheet_name", default=None, help="Sheet name (optional). If omitted, uses first sheet.")
    ap.add_argument("--provider", default="gemini", choices=["gemini"], help="VLM provider (currently gemini).")
    ap.add_argument("--mode", default="vertex", choices=["vertex", "api"], help="Gemini via Vertex or direct API key.")
    ap.add_argument("--model", default="gemini-2.0-pro-exp-02-05", help="Model name.")
    ap.add_argument("--project", default=None, help="GCP project (Vertex).")
    ap.add_argument("--location", default="us-central1", help="GCP location (Vertex).")
    args = ap.parse_args()

    # Load & convert image
    img_bytes = load_image_bytes(args.image_or_pdf)

    # Build strict prompt
    prompt = build_prompt()

    # Call VLM
    try:
        if args.provider == "gemini" and args.mode == "vertex":
            if not args.project:
                raise RuntimeError("--project is required for Vertex mode")
            raw = call_gemini_vertex(img_bytes, prompt, model=args.model, project=args.project, location=args.location)
        else:
            raw = call_gemini_api(img_bytes, prompt, model=args.model)
    except Exception as e:
        print(f"[WARN] Model call failed: {e}\nUsing heuristic fallback...", file=sys.stderr)
        df_normalized = heuristic_fallback(args.rr_path, args.sheet_name)
        print(df_normalized.head())
        return

    code = extract_python_block(raw)
    if not code:
        print("[WARN] No code block found in model response. Using heuristic fallback...", file=sys.stderr)
        df_normalized = heuristic_fallback(args.rr_path, args.sheet_name)
        print(df_normalized.head())
        return

    # Execute generated code with injected variables
    try:
        locs = safe_exec(code, rr_path=args.rr_path, sheet_name=args.sheet_name)
        if "df_normalized" not in locs:
            raise RuntimeError("Model code did not produce df_normalized.")
        df_normalized = locs["df_normalized"]
        # Display a small preview
        try:
            print(df_normalized.head())
        except Exception:
            pass
    except Exception as e:
        print(f"[WARN] Execution error: {e}\nUsing heuristic fallback...", file=sys.stderr)
        df_normalized = heuristic_fallback(args.rr_path, args.sheet_name)
        print(df_normalized.head())

if __name__ == "__main__":
    main()

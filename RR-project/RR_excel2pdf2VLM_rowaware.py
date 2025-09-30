#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rent Roll Column Extraction (Excel -> PDF chunks -> Gemini VLM -> JSON) + Stitching
- REST call to Vertex AI "generateContent" (Gemini VLM)
- pdf2image for PDF->PIL images
- Parallel per-page calls
- Schema-aware prompts for rent-roll columns; handles columns split across pages
- Consolidates per-page outputs into a single table with provenance

Env:
  export GEMINI_API_KEY="ya29...."    # or fetch via your vault before launching
  export GEMINI_ENDPOINT="https://us-central-aiplatform.googleapis.com/v1/projects/<proj>/locations/us-central1/publishers/google/models/gemini-2.5-flash:generateContent"

Run:
  python rentroll_extractor.py rent_roll.pdf \
      --columns "Unit,Address,Resident,SQFT,Status,Lease Start,Lease End,Base Rent,Other Charges,Total Monthly" \
      --hints_file hints.json \
      --out consolidated.json
"""

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

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_ENDPOINT = os.environ.get("GEMINI_ENDPOINT", "").strip()
# Example:
#   export GEMINI_ENDPOINT="https://us-central-aiplatform.googleapis.com/v1/projects/<proj>/locations/us-central1/publishers/google/models/gemini-2.5-flash:generateContent"

if not GEMINI_API_KEY or not GEMINI_ENDPOINT:
    raise RuntimeError("Set GEMINI_API_KEY and GEMINI_ENDPOINT (full :generateContent URL).")

MAX_WORKERS = max(1, min(8, os.cpu_count() or 4))

# Default hints (can override via --hints_file)
DEFAULT_HINTS = {
    "columns": {
        "Unit":          {"aliases": ["Unit #", "Apt", "Apartment", "Unit Number"], "section_hints": ["Rent Roll"], "notes": ""},
        "Address":       {"aliases": ["Street", "Location"], "section_hints": ["Rent Roll"], "notes": ""},
        "Resident":      {"aliases": ["Tenant", "Occupant", "Lessee"], "section_hints": ["Rent Roll"], "notes": ""},
        "SQFT":          {"aliases": ["Area", "Square Feet", "SF"], "section_hints": ["Rent Roll"], "notes": ""},
        "Status":        {"aliases": ["Occ. Status", "Occupancy", "Unit Status"], "section_hints": ["Rent Roll"], "notes": ""},
        "Lease Start":   {"aliases": ["Start Date", "Lease Start Date"], "section_hints": ["Lease Terms"], "notes": ""},
        "Lease End":     {"aliases": ["End Date", "Lease End Date"], "section_hints": ["Lease Terms"], "notes": ""},
        "Base Rent":     {"aliases": ["Monthly Rent", "Contract Rent", "Rent"], "section_hints": ["Charges", "Rent"], "notes": "currency"},
        "Other Charges": {"aliases": ["Add'l Charges", "Fees"], "section_hints": ["Charges"], "notes": "currency"},
        "Total Monthly": {"aliases": ["Total", "Total Monthly Charges"], "section_hints": ["Charges"], "notes": "sum row ok"},
    },
    "table_hints": {
        "titles": ["Rent Roll", "Unit Schedule", "Tenant Roster"],
        "sections": ["Rent Roll", "Lease Terms", "Charges"],
        "line_items": ["Total", "Subtotal", "Aggregate"],
        "layout_notes": "Expect boxed regions; headers may repeat per page; columns may continue on next page."
    }
}

GENERATION_CONFIG = {
    "temperature": 0.0,
    "topP": 1.0,
    "topK": 40,
    "maxOutputTokens": 1536,
    "responseMimeType": "application/json",
}


# ============================================================
# 1) PDF -> IMAGES
# ============================================================

def pdf_to_images(pdf_path: str, dpi: int = 220) -> List[Image.Image]:
    """
    Convert each PDF page to a PIL Image.
    NOTE: Install Poppler (pdf2image dependency).
    """
    return convert_from_path(pdf_path, dpi=dpi)


# ============================================================
# 2) PROMPTS (schema-aware for columns across chunks)
# ============================================================


def build_system_instructions() -> str:
    """
    System message for extracting table columns from rent roll printed to multi-page PDF chunks.
    UPDATED: Enforce capturing a leading "Row Number" column and return it per row to enable order-aware stitching.
    """
    return """You are a document vision model specialized in extracting tables from Excel sheets that were printed to multi-page PDFs.
The source is a RENT ROLL worksheet printed with gridlines and **a leading column labeled 'Row Number'**; the PDF pages are CHUNKS of the same sheet.

GOAL (per page): Extract the requested COLUMNS of the rent roll as structured JSON for THIS page.
Rows may span across multiple PDF pages; columns may be split across pages. The caller will stitch pages using 'row_number' primarily, and secondarily by primary keys.

Key rules:
1) TABLE & HEADERS
   - Detect the main rent roll table area on the page.
   - Read/align the column headers as printed on this page; headers may repeat per chunk and may be partially visible.
   - The first column is **Row Number**; ALWAYS read it and return it as an integer field 'row_number' for each row you output.

2) COLUMNS TO RETURN
   - Return ONLY the columns requested by the user prompt (exact names) inside the 'values' object.
   - Do NOT normalize names; hints/aliases are only cues to locate columns. Keep the user's column names as-is.
   - If a requested column's cell is not visible on this page/chunk, set its value to null.
   - You MUST still emit a row with its 'row_number' whenever that row appears on this page, even if many requested values are null.

3) ROW GRANULARITY
   - Output one JSON row PER PRINTED ROW in the table body that is visible on this page.
   - Preserve the original order using 'row_number'. Do not invent or skip row numbers.
   - For multi-line/unit blocks (e.g., charge schedules, rent steps) that span multiple rows, emit each physical row with its own 'row_number'.
   - DO NOT aggregate rows into units; aggregation is done later.

4) PROVENANCE
   - For each returned cell, include a provenance object with either the Excel-like cell coordinate if legible (e.g., "N23") or a bounding box [x0,y0,x1,y1] in page pixels; if unknown, use null.

5) OUTPUT FORMAT (STRICT JSON; NO MARKDOWN FENCES)
Return a JSON object for THIS PAGE ONLY:
{
  "page_index": <int, 0-based>,
  "columns_visible": ["<subset of requested columns visible on this page (left-to-right)>"],
  "rows": [
    {
      "row_number": <int>,   // REQUIRED
      "row_key": "<best identifier for this row on this page (e.g., Unit/Address) or empty string>",
      "values": {
        "<RequestedColumnName1>": "<cell text or null>",
        "<RequestedColumnName2>": ...
      },
      "provenance": {
        "<RequestedColumnName1>": { "cell_coordinate": "N23" or null, "bbox": [x0,y0,x1,y1] or null },
        "<RequestedColumnName2>": { ... }
      }
    },
    ...
  ]
}
Return JSON only, no prose.
"""


def build_user_prompt(
    page_index: int,
    target_columns: List[str],
    hints: Dict[str, Any],
    primary_keys: List[str],
) -> str:
    """
    Page-specific user prompt for rent-roll column extraction.
    UPDATED: Always capture and return the 'row_number' from the leading 'Row Number' column.
    - target_columns: exact column names the caller wants inside "values"
    - hints: variability cues (columns aliases/sections etc.)
    - primary_keys: stable identifiers (e.g., ["Unit"] or ["Unit","Address"]) used only as tiebreakers in stitching
    """
    # We do not add "Row Number" into target_columns; model must still return 'row_number' as top-level field.
    payload = {
        "context": {
            "document": "Single Excel rent roll printed across multiple PDF pages (Z-order).",
            "page_index": page_index,
            "notes": [
                "The first column in the sheet is 'Row Number'—return it as integer 'row_number' for each emitted row.",
                "Rows may span multiple pages; return each physical row that is visible on this page, even if values are null.",
            ],
        },
        "requested_columns": target_columns,
        "stitching": {
            "primary_keys": primary_keys,
            "ordering": "Sort by row_number when consolidating; primary_keys only as secondary identifiers.",
        },
        "hints": hints,
        "instructions": "Return STRICT JSON only as specified by the system message."
    }
    return json.dumps(payload, ensure_ascii=False)


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

    resp = requests.post(GEMINI_ENDPOINT, headers=headers, json=request_body, timeout=180)
    resp.raise_for_status()
    data = resp.json()

    # Typical Vertex response shape: candidates[0].content.parts[0].text
    candidates = data.get("candidates", [])
    parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
    text = parts[0].get("text", "") if parts else ""
    return text or json.dumps(data)


# ============================================================
# 4) PAGE RESULT MODEL + CALLER
# ============================================================

@dataclass
class PageResult:
    page_index: int
    columns_visible: List[str]
    rows: List[Dict[str, Any]]
    notes: Optional[str]
    error: Optional[str] = None


def call_llm_on_page(
    img: Image.Image,
    page_index: int,
    target_columns: List[str],
    hints: Dict[str, Any],
    primary_keys: List[str],
    system_instructions: str
) -> PageResult:
    try:
        user_prompt = build_user_prompt(page_index, target_columns, hints, primary_keys)
        text = (extract_from_image(img, user_prompt, system_instructions) or "").strip()

        # strip accidental code fences
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("Non-dict JSON returned")

        # Minimal validation
        if data.get("page_index") != page_index:
            # don't fail hard—accept but record a note
            data["notes"] = (data.get("notes") or "") + " [warning: page_index mismatch]"

        return PageResult(
            page_index=page_index,
            columns_visible=list(data.get("columns_visible", [])),
            rows=list(data.get("rows", [])),
            notes=data.get("notes"),
        )
    except Exception as e:
        return PageResult(page_index=page_index, columns_visible=[], rows=[], notes=None, error=str(e))


# ============================================================
# 5) STITCHING (merge complementary chunks)
# ============================================================

def _is_better_value(new_val: Any, old_val: Any) -> bool:
    """
    Tie-breaker when two non-null values appear for the same row+column from different pages.
    Heuristics: prefer numeric-looking over non-numeric; otherwise keep newest (caller order).
    """
    if old_val is None:
        return True

    def _numlike(s):
        try:
            float(str(s).replace(",", "").replace("$", ""))
            return True
        except Exception:
            return False

    if _numlike(new_val) and not _numlike(old_val):
        return True
    return False  # else last-write-wins via caller order



def stitch_pages(
    per_page: List[PageResult],
    target_columns: List[str],
    primary_keys: List[str]
) -> Dict[str, Any]:
    """
    Consolidate per-page JSONs into a single, row-order–aware table.
    Strategy:
      - Primary key for stitching is 'row_number' (monotone increasing down the sheet, unique per printed row).
      - If 'row_number' is missing for a row from the model, fall back to a hash of primary_keys ('row_key') for that row.
      - Merge values across chunks by coalescing non-null fields; keep first non-empty provenance unless a later chunk has a non-null where previous was null.
      - Sort final rows by 'row_number' (ascending), with any fallbacks (no row_number) appended in encounter order.
    """
    # Map of row_id -> aggregate where row_id is int(row_number) when present, else a string fallback
    aggregated: Dict[typing.Union[int, str], Dict[str, Any]] = {}
    encounter_counter = 0  # for stable order of no-row-number fallbacks

    def coalesce(dst: Dict[str, Any], src: Dict[str, Any]):
        # Merge "values"
        for col in target_columns:
            sv = src.get("values", {}).get(col, None)
            dv = dst["values"].get(col, None)
            if dv in (None, "", []) and sv not in (None, "", []):
                dst["values"][col] = sv
        # Merge provenance per column
        for col in target_columns:
            sp = src.get("provenance", {}).get(col, None)
            dp = dst["provenance"].get(col, None)
            if (dp is None or all(v is None for v in (dp.get("cell_coordinate") if isinstance(dp, dict) else [None]))) and sp is not None:
                dst["provenance"][col] = sp
        # Track pages seen
        dst["pages_seen"].extend(src.get("pages_seen", []))

    for page in per_page:
        if page.error:
            continue
        for r in page.rows:
            rn = r.get("row_number", None)
            # Compose a fallback id if rn missing
            if rn is None:
                # Use the provided row_key if present, else build from primary key values present in r['values']
                row_key = r.get("row_key", "")
                if not row_key:
                    pk_vals = []
                    for pk in primary_keys:
                        pk_vals.append(str(r.get("values", {}).get(pk, "")))
                    row_key = "|".join(pk_vals) if any(pk_vals) else f"__noid__{encounter_counter}"
                    encounter_counter += 1
                row_id = row_key
            else:
                try:
                    row_id = int(str(rn).strip())
                except Exception:
                    row_id = f"__bad_rownum__{str(rn)}"

            if row_id not in aggregated:
                aggregated[row_id] = {
                    "row_number": rn if rn is not None and isinstance(rn, (int, float, str)) else None,
                    "row_key": r.get("row_key", ""),
                    "values": {col: None for col in target_columns},
                    "provenance": {col: None for col in target_columns},
                    "pages_seen": []
                }
            # Build a src-like dict with pages_seen info
            src_row = {
                "values": r.get("values", {}),
                "provenance": r.get("provenance", {}),
                "pages_seen": [page.page_index],
            }
            coalesce(aggregated[row_id], src_row)

    # Prepare final rows: sort by row_number where available, keep fallbacks in insertion order after numbered ones
    numbered = [(rid, data) for rid, data in aggregated.items() if isinstance(rid, int)]
    unnumbered = [(rid, data) for rid, data in aggregated.items() if not isinstance(rid, int)]

    numbered.sort(key=lambda x: x[0])
    # unnumbered order preserved by dict insertion in Python 3.7+; we leave as-is

    consolidated_rows = []
    for rid, data in numbered + unnumbered:
        # Deduplicate pages_seen and sort
        data["pages_seen"] = sorted(set(data["pages_seen"]))
        consolidated_rows.append({
            "row_number": data.get("row_number"),
            "row_key": data.get("row_key", ""),
            "values": data.get("values", {}),
            "provenance": data.get("provenance", {}),
            "pages_seen": data["pages_seen"]
        })

    return {
        "columns": target_columns,
        "rows": consolidated_rows
    }


def extract_rentroll_columns_from_pdf(
    pdf_path: str,
    target_columns: List[str],
    hints: Dict[str, Any],
    primary_keys: List[str],
    dpi: int = 220,
    max_workers: int = MAX_WORKERS,
) -> Dict[str, Any]:
    """
    End-to-end:
      PDF -> images -> parallel VLM calls -> per-page JSONs -> stitch -> return consolidated.
    """
    images = pdf_to_images(pdf_path, dpi=dpi)
    system_instructions = build_system_instructions()

    per_page: List[PageResult] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(
                call_llm_on_page,
                img,
                idx,
                target_columns,
                hints,
                primary_keys,
                system_instructions
            ): idx
            for idx, img in enumerate(images)
        }
        for fut in as_completed(futures):
            per_page.append(fut.result())

    consolidated = stitch_pages(per_page, target_columns, primary_keys)

    return {
        "document": os.path.basename(pdf_path),
        "target_columns": target_columns,
        "primary_keys": primary_keys,
        "pages_processed": len(images),
        "per_page": [p.__dict__ for p in sorted(per_page, key=lambda x: x.page_index)],
        "consolidated": consolidated
    }


# ============================================================
# 7) CLI
# ============================================================

def parse_columns_arg(arg: str) -> List[str]:
    # allow comma-separated or JSON array
    arg = arg.strip()
    if arg.startswith("["):
        return json.loads(arg)
    return [c.strip() for c in arg.split(",") if c.strip()]

def main():
    p = argparse.ArgumentParser(description="Extract rent-roll columns from a printed-to-PDF worksheet using Gemini VLM (REST).")
    p.add_argument("pdf_path", help="Path to the multi-page PDF (printed Excel rent roll).")
    p.add_argument("--columns", required=True,
                   help="Requested columns (comma-separated or JSON array). "
                        "Example: \"Unit,Address,Resident,SQFT,Status,Lease Start,Lease End,Base Rent,Other Charges,Total Monthly\"")
    p.add_argument("--primary_keys", default="Unit",
                   help="Primary keys for stitching rows (comma-separated or JSON array). Example: \"Unit,Address\"")
    p.add_argument("--hints_file", default="",
                   help="Optional JSON file with hints dict (overrides defaults).")
    p.add_argument("--dpi", type=int, default=220, help="Rendering DPI for pdf2image (higher -> sharper).")
    p.add_argument("--workers", type=int, default=MAX_WORKERS, help="Max parallel LLM calls.")
    p.add_argument("--out", default="", help="Path to write the final consolidated JSON.")
    args = p.parse_args()

    target_columns = parse_columns_arg(args.columns)
    primary_keys = parse_columns_arg(args.primary_keys)

    hints = DEFAULT_HINTS
    if args.hints_file:
        with open(args.hints_file, "r", encoding="utf-8") as f:
            hints = json.load(f)

    result = extract_rentroll_columns_from_pdf(
        pdf_path=args.pdf_path,
        target_columns=target_columns,
        hints=hints,
        primary_keys=primary_keys,
        dpi=args.dpi,
        max_workers=args.workers,
    )

    out_json = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out_json)
        print(f"Wrote {args.out}")
    else:
        print(out_json)


if __name__ == "__main__":
    main()

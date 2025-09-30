#Below are drop-in replacements for build_system_instructions() and build_user_prompt(...) plus a tiny example of how to pass your target columns and hints.
#1) Replace your build_system_instructions() with this


def build_system_instructions() -> str:
    """
    System message for extracting entire table columns from a rent roll
    printed to multi-page PDF chunks. Columns may be split across pages.
    Hints are variability cues only (not normalization rules).
    """
    return """\
You are a document vision model specialized in extracting tables from Excel sheets that were printed to multi-page PDFs.
The source is a RENT ROLL worksheet printed with gridlines and row/column headers; the PDF pages are CHUNKS of the same sheet.

GOAL (per entire document, returned per page): Extract the requested COLUMNS of the rent roll as structured JSON.
Rows may span across multiple PDF pages; columns may be split across chunks. Reconstruct the table consistently.

Key rules:
1) TABLE RECOGNITION & HEADERS
   - Detect the main rent roll table on each page.
   - Read/align column headers as printed on that page; headers may repeat per chunk.
   - Column names may vary; use provided hints as search cues only (NOT for renaming or normalization).
   - Treat multi-row headers (e.g., split across two lines) as a single header.

2) ROW CONTINUITY ACROSS CHUNKS
   - Rows continue across pages. Use stable row identifiers (e.g., Unit, Address, or an explicit primary key if supplied) to stitch rows.
   - If a row is cut at a page boundary, continue it on the next page chunk.
   - Do NOT duplicate rows that reappear due to header repetition.

3) COLUMN CONTINUITY ACROSS CHUNKS
   - Requested columns may be partially visible on one page and continue on the next.
   - If a requested column is not visible on a page, mark values as missing for that page; resume when the column reappears.
   - When columns appear under slightly different header text, rely on hints to decide it is the same logical column.

4) CELL COORDINATES & EVIDENCE
   - For each extracted cell, record the page_index and a best-effort Excel-like cell coordinate (e.g., N23) using visible row/column headers.
   - Provide an approximate pixel bbox on that page for provenance.

5) OUTPUT FORMAT (STRICT JSON; NO MARKDOWN FENCES)
Return for EACH page processed a JSON object with this schema; the caller will aggregate pages:
{
  "page_index": <int, 0-based>,
  "columns_visible": ["<subset of requested columns visible on this page in left-to-right order>"],
  "rows": [
    {
      "row_key": "<best unique identifier for the row on this page (e.g., Unit/Address/Unit+Address)>",
      "values": {
        "<RequestedColumnName1>": "<cell text or null if not visible on this page>",
        "<RequestedColumnName2>": "...",
        ...
      },
      "provenance": {
        "<RequestedColumnName1>": {
          "cell_coordinate": "N23" or null,
          "bbox": [x0,y0,x1,y1] or null
        },
        ...
      }
    },
    ...
  ],
  "notes": "<brief remarks if a row or column continues on next/prev chunk>"
}

Confidence policy:
- Be conservative when headers are ambiguous; prefer leaving a cell null over guessing.
- Use hints only as signals of variability in names and layout, NOT as normalization targets.

Return JSON only, no prose.
"""

#2) Replace your build_user_prompt(...) with this

def build_user_prompt(
    page_index: int,
    target_columns: list,
    hints: dict,
    primary_keys: list = None,
) -> str:
    """
    Build a page-specific user prompt for rent-roll column extraction.
    - target_columns: exact column names you want in the output JSON
    - hints: {
        "columns": { "<RequestedColumnName>": { "aliases": [...], "section_hints": [...], "notes": "..." }, ... },
        "table_hints": { "titles": [...], "sections": [...], "line_items": [...], "layout_notes": "..." }
      }
      All hints are variability cues ONLY (not normalization rules).
    - primary_keys: preferred row identifiers, e.g., ["Unit", "Address"] to help stitch across pages
    """
    primary_keys = primary_keys or ["Unit", "Address"]

    payload = {
        "context": {
            "document": "Single Excel rent roll printed across multiple PDF pages (Z-order).",
            "page_index": page_index,
            "stitching": {
                "rows_across_pages_by": primary_keys,
                "columns_may_split_across_pages": True
            }
        },
        "request": {
            "target_columns": target_columns,
            "primary_keys": primary_keys
        },
        "hints": hints,  # variability cues only
        "instructions": "Extract ONLY the requested columns into the required JSON schema for THIS page; do not include extra columns."
    }

    return (
        "RENT ROLL COLUMN EXTRACTION REQUEST\n"
        "Follow the System Instructions. This prompt contains page context, requested columns, and variability hints.\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )

# 3) How to call it (example)
# Use the same pipeline as before. Just pass target_columns and hints when you build the per-page prompt:

# Example targets you want extracted end-to-end
TARGET_COLUMNS = [
    "Unit", "Address", "Resident", "SQFT",
    "Status", "Lease Start", "Lease End", "Base Rent", "Other Charges", "Total Monthly"
]

# Example variability cues (use your real ones)
RENT_ROLL_HINTS = {
    "columns": {
        "Unit":          {"aliases": ["Unit #", "Apt", "Apartment", "Unit Number"], "section_hints": ["Rent Roll", "Units"], "notes": ""},
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

# When you make the per-page call:
system_instructions = build_system_instructions()
user_prompt = build_user_prompt(
    page_index=idx,
    target_columns=TARGET_COLUMNS,
    hints=RENT_ROLL_HINTS,
    primary_keys=["Unit"]  # or ["Unit","Address"] if you have both
)
# ... then call extract_from_image(img, user_prompt, system_instructions) as before

# What this achieves
# Treats columns and rows as a stitched table across chunks.
# Uses your hints purely as variability signals to locate columns/sections, not to rename.
# Outputs strict JSON per page that your existing aggregator can merge into a single table (since each row has a row_key, per-cell provenance, and we track which columns are visible on each page).

# Here’s a compact, drop-in stitcher you can use right after you collect the per-page results (works with the JSON shape we defined):

from typing import List, Dict, Any, Tuple
import math

def _is_better_value(new_val: str, old_val: str, new_conf: float, old_conf: float) -> bool:
    """Tie-breaker for conflicting non-null values."""
    # 1) higher confidence wins
    if (new_conf or 0) > (old_conf or 0) + 1e-9:
        return True
    if abs((new_conf or 0) - (old_conf or 0)) <= 1e-9:
        # 2) prefer numeric-looking when both present
        def _numlike(s):
            try:
                float(str(s).replace(",", "").replace("$",""))
                return True
            except Exception:
                return False
        if _numlike(new_val) and not _numlike(old_val):
            return True
        # 3) else let caller’s iteration order (typically page order) decide: last-write-wins
    return False

def stitch_rentroll_pages(
    per_page_objects: List[Dict[str, Any]],
    target_columns: List[str],
) -> Dict[str, Any]:
    """
    per_page_objects: list of page JSONs emitted by the VLM (our per-page schema)
    target_columns: ordered list of requested columns
    Returns a consolidated table + a flat audit.
    """
    master: Dict[str, Dict[str, Any]] = {}  # row_key -> {values, provenance}
    # Keep simple page-level confidence map (fallback 0.0)
    page_conf: Dict[int, float] = {
        obj.get("page_index", -1): (obj.get("confidence", 0.0) if isinstance(obj.get("confidence", None), (int,float)) else 0.0)
        for obj in per_page_objects
    }

    for page in per_page_objects:
        if not isinstance(page, dict):
            continue
        page_idx = page.get("page_index", -1)
        rows = page.get("rows", [])
        for r in rows:
            row_key = r.get("row_key")
            if not row_key:
                continue
            # Ensure master row exists
            if row_key not in master:
                master[row_key] = {
                    "values": {col: None for col in target_columns},
                    "provenance": {col: None for col in target_columns},
                    "sightings": []  # optional: keep every page where the row appeared
                }
            m = master[row_key]
            m["sightings"].append(page_idx)

            # Merge each requested column
            vals: Dict[str, Any] = r.get("values", {})
            prov: Dict[str, Any] = r.get("provenance", {})
            for col in target_columns:
                new_val = vals.get(col, None)
                if new_val is None:
                    continue  # leave as-is; another chunk may fill it
                old_val = m["values"].get(col, None)
                if old_val is None or _is_better_value(new_val, old_val, page_conf.get(page_idx, 0.0), 0.0):
                    m["values"][col] = new_val
                    m["provenance"][col] = prov.get(col, None)

    # Convert to ordered list of rows (stable by row_key)
    consolidated_rows = []
    for row_key in sorted(master.keys()):
        consolidated_rows.append({
            "row_key": row_key,
            "values": master[row_key]["values"],
            "provenance": master[row_key]["provenance"],
            "pages_seen": sorted(set(master[row_key]["sightings"]))
        })

    return {
        "columns": target_columns,
        "rows": consolidated_rows
    }




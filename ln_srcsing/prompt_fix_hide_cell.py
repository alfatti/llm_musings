# one helper to build the rectangle, and a new vicinity_desc block that only talks about a rectangle centered on the target cell (no "D17" or similar).

from typing import Dict, Any, Optional
import re


def parse_cell_hint(cell_hint: Optional[str]):
    """
    Parses a cell hint like 'D17' → (row=17, col_idx=4).
    Returns None if the hint is missing or malformed.
    """
    if cell_hint is None:
        return None

    # treat several forms as missing
    if str(cell_hint).strip().upper() in {"", "N/A", "NA", "NONE", "NULL"}:
        return None

    m = re.match(r"([A-Za-z]+)(\d+)$", cell_hint.strip())
    if not m:
        return None

    col_letters, row_str = m.groups()
    row = int(row_str)

    # Convert column letters → number
    col_idx = 0
    for ch in col_letters.upper():
        col_idx = col_idx * 26 + (ord(ch) - ord("A") + 1)

    return row, col_idx



def build_vicinity_desc(
    cell_hint: Optional[str],
    vicinity_rows: int,
    vicinity_cols: int,
) -> Dict[str, Any]:
    """
    Build a vicinity description for the LLM.
    If cell_hint cannot be parsed → return a block instructing the model
    to return NULL rather than guess.
    """

    parsed = parse_cell_hint(cell_hint)

    # --------------------------------------------------------
    # CASE 1 — No valid hint → enforce NULL extraction
    # --------------------------------------------------------
    if parsed is None:
        return {
            "rectangle_center": None,
            "rectangle_bounds": None,
            "instructions": (
                "The target cell could not be reliably located because no valid row/column "
                "coordinates were provided. Do NOT guess or infer a value from the page. "
                "Return both the extracted value and its location as NULL."
            ),
        }

    # --------------------------------------------------------
    # CASE 2 — Valid hint → compute numeric rectangle
    # --------------------------------------------------------
    hint_row, hint_col_idx = parsed

    row_start = max(1, hint_row - vicinity_rows)
    row_end   = hint_row + vicinity_rows
    col_start = max(1, hint_col_idx - vicinity_cols)
    col_end   = hint_col_idx + vicinity_cols

    narrative = (
        "Use a vertical-first scanning strategy. The target cell is represented only by "
        "numeric row/column indices (no Excel address). Treat this point as the center of "
        "a rectangular neighborhood. Prioritize values located within this rectangle. "
        "If no valid value appears inside this rectangle, return NULL rather than guessing."
    )

    return {
        "rectangle_center": {
            "row_index": hint_row,
            "column_index_1_based": hint_col_idx,
        },
        "rectangle_bounds": {
            "row_start_index": row_start,
            "row_end_index": row_end,
            "col_start_index_1_based": col_start,
            "col_end_index_1_based": col_end,
        },
        "instructions": narrative,
    }


# Then in your main prompt-building code you just do:

vicinity_desc = build_vicinity_desc(
    hint_row=hint_row,
    hint_col_idx_1_based=hint_col_idx,
    vicinity_rows=vicinity_rows,
    vicinity_cols=vicinity_cols,
)



# one helper to build the rectangle, and a new vicinity_desc block that only talks about a rectangle centered on the target cell (no "D17" or similar).

from typing import Dict, Any

import re

def parse_cell_hint(cell_hint: str):
    """
    Example: 'D17' -> (row=17, col_idx=4)
    """
    m = re.match(r"([A-Za-z]+)(\d+)", cell_hint)
    if not m:
        raise ValueError(f"Invalid cell hint: {cell_hint}")

    col_letters, row_str = m.groups()
    row = int(row_str)

    # Convert column letters → number (A=1, B=2, ..., Z=26, AA=27, etc.)
    col_idx = letters_to_number(col_letters)

    return row, col_idx


def letters_to_number(col_letters: str) -> int:
    """
    Excel-style column letters to 1-based index.
    """
    col_letters = col_letters.upper()
    num = 0
    for ch in col_letters:
        num = num * 26 + (ord(ch) - ord("A") + 1)
    return num


def build_vicinity_desc(
    hint_row: int,
    hint_col_idx_1_based: int,
    vicinity_rows: int,
    vicinity_cols: int,
) -> Dict[str, Any]:
    """
    Build a vicinity description without exposing the raw cell address (like 'D17').
    The LLM is told to focus on a rectangular neighborhood centered on the target cell.
    """

    # Compute numeric bounds of the rectangle around the target cell
    row_start = max(1, hint_row - vicinity_rows)
    row_end = hint_row + vicinity_rows

    col_start = max(1, hint_col_idx_1_based - vicinity_cols)
    col_end = hint_col_idx_1_based + vicinity_cols

    narrative = (
        "Use a vertical-first scanning strategy. Treat the target cell as the center of a "
        "rectangular neighborhood: look up to {rows_up} rows above and {rows_down} rows below "
        "the target row, and up to {cols_left} columns to the left and {cols_right} columns "
        "to the right of the target column. Prioritize matches inside this rectangle. "
        "If no suitable match is found inside this rectangle, return the value and location "
        "as NULL rather than guessing."
    ).format(
        rows_up=vicinity_rows,
        rows_down=vicinity_rows,
        cols_left=vicinity_cols,
        cols_right=vicinity_cols,
    )

    vicinity_desc: Dict[str, Any] = {
        "bias": "vertical-first",
        "rectangle_center": {
            # Numeric indices only; no human-readable cell label
            "row_index": hint_row,
            "column_index_1_based": hint_col_idx_1_based,
        },
        "rectangle_half_extents": {
            "rows_above": vicinity_rows,
            "rows_below": vicinity_rows,
            "cols_left": vicinity_cols,
            "cols_right": vicinity_cols,
        },
        "rectangle_bounds": {
            "row_start_index": row_start,
            "row_end_index": row_end,
            "col_start_index_1_based": col_start,
            "col_end_index_1_based": col_end,
        },
        "instructions": narrative,
    }

    return vicinity_desc

# Then in your main prompt-building code you just do:

vicinity_desc = build_vicinity_desc(
    hint_row=hint_row,
    hint_col_idx_1_based=hint_col_idx,
    vicinity_rows=vicinity_rows,
    vicinity_cols=vicinity_cols,
)



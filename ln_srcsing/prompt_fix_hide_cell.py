from typing import Dict, Any, Optional

def build_vicinity_desc(
    cell_hint: Optional[str],
    hint_row: Optional[int],
    hint_col_idx_1_based: Optional[int],
    vicinity_rows: int,
    vicinity_cols: int,
) -> Dict[str, Any]:
    """
    Build a vicinity description for the prompt.

    If cell_hint is missing (None, empty, 'N/A', etc.), return a guard block that signals
    no geometric prior exists. The LLM must not guess a location in that case.
    """

    # -------------------------
    # CASE 1 — No valid hint
    # -------------------------
    if not cell_hint or str(cell_hint).strip().upper() in {"N/A", "NA", "NONE", ""}:
        return {
            "bias": "no-location-prior",
            "instructions": (
                "No location hint is available for this variable. "
                "Do not infer or guess any row/column coordinates. "
                "Search normally, but if the variable cannot be clearly identified "
                "with high confidence, set both value=NULL and location=NULL."
            ),
            "rectangle_center": None,
            "rectangle_half_extents": None,
            "rectangle_bounds": None,
        }

    # ------------------------------------
    # CASE 2 — Valid hint given numerically
    # (we assume the caller already parsed
    #   hint_row/hint_col_idx safely)
    # ------------------------------------

    # Compute numeric bounds of the rectangle around the target cell
    row_start = max(1, hint_row - vicinity_rows)
    row_end = hint_row + vicinity_rows

    col_start = max(1, hint_col_idx_1_based - vicinity_cols)
    col_end = hint_col_idx_1_based + vicinity_cols

    narrative = (
        "Use a vertical-first scanning strategy. Treat the target cell as the center of a "
        "rectangular neighborhood: inspect up to {rows_up} rows above and {rows_down} rows below "
        "the target row, and up to {cols_left} columns to the left and {cols_right} columns "
        "to the right of the target column. Prioritize matches inside this rectangle. "
        "If no suitable match exists inside this rectangle, return value=NULL and location=NULL "
        "rather than guessing."
    ).format(
        rows_up=vicinity_rows,
        rows_down=vicinity_rows,
        cols_left=vicinity_cols,
        cols_right=vicinity_cols,
    )

    return {
        "bias": "vertical-first",
        "rectangle_center": {
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

# Then in your main prompt-building code you just do:

vicinity_desc = build_vicinity_desc(
    hint_row=hint_row,
    hint_col_idx_1_based=hint_col_idx,
    vicinity_rows=vicinity_rows,
    vicinity_cols=vicinity_cols,
)



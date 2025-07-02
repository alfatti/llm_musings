import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Any

# ───── helper canon + reason unchanged ─────
# ... _canonicalise() and _reason() here ...

def compare_csv_to_excel_tab(csv_path: str,
                             xlsx_path: str,
                             target_tab: str,
                             *,                     # force keyword args from here on
                             excel_header_row: int = 9,   # <-- NEW
                             numeric_atol: float = 1e-9,
                             verbose: bool = True
                            ) -> Tuple[bool, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Compare a CSV extract to a specific Excel sheet whose *true*
    headers start on `excel_header_row` (0-based).
    """
    # 1. Load CSV (header row 0 as usual)
    df_csv = pd.read_csv(csv_path)

    # 2. Load Excel tab, telling pandas where the real header is.
    #    skiprows=range(excel_header_row) discards the 'decorative' rows.
    df_excel = pd.read_excel(
        xlsx_path,
        sheet_name=target_tab,
        header=excel_header_row,
        skiprows=range(excel_header_row)
    )

    # 3. Normalise column names (unchanged)  ──────────────────────────
    df_csv.columns   = df_csv.columns.str.strip().str.casefold()
    df_excel.columns = df_excel.columns.str.strip().str.casefold()
    shared_cols = df_csv.columns.intersection(df_excel.columns)
    if shared_cols.empty:
        raise ValueError("No shared columns to compare.")

    # 4. Align lengths / continue as before  ─────────────────────────
    n = min(len(df_csv), len(df_excel))
    df_csv, df_excel = df_csv.iloc[:n].reset_index(drop=True), df_excel.iloc[:n].reset_index(drop=True)

    summary_rows: List[Dict[str, Any]] = []
    detail_dict : Dict[str, pd.DataFrame] = {}
    overall_ok  = True

    for col in shared_cols:
        s1, tag1 = _canonicalise(df_csv[col])
        s2, tag2 = _canonicalise(df_excel[col])
        tag = tag1 if tag1 == tag2 else "mixed"

        if tag == "numeric":
            equal_mask = np.isclose(s1, s2, atol=numeric_atol, equal_nan=True)
        elif tag == "datetime":
            equal_mask = s1.eq(s2) | (s1.isna() & s2.isna())
        else:
            equal_mask = s1.eq(s2) | (s1.isna() & s2.isna())

        mismatches = (~equal_mask).sum()
        summary_rows.append(dict(column=col, dtype=tag, passed=mismatches == 0, mismatches=mismatches))

        if mismatches:
            overall_ok = False
            detail_dict[col] = pd.DataFrame({
                "row"   : np.flatnonzero(~equal_mask),
                "csv"   : s1[~equal_mask].values,
                "excel" : s2[~equal_mask].values,
                "reason": [
                    _reason(tag, v1, v2, numeric_atol)
                    for v1, v2 in zip(s1[~equal_mask], s2[~equal_mask])
                ]
            })

    summary_df = pd.DataFrame(summary_rows)

    if verbose:
        from IPython.display import display
        print("\n=== Column-by-column summary ===")
        display(summary_df)
        for col, df in detail_dict.items():
            print(f"\n❌ Detail for column '{col}'")
            display(df)
        if overall_ok:
            print("\n✅ All shared columns match after normalisation.")

    return overall_ok, summary_df, detail_dict

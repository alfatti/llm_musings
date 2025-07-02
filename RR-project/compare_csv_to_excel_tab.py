import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Any

# ─────────────────────────────────────────────────────────
# 0.  Helpers
# ─────────────────────────────────────────────────────────
def _canonicalise(s: pd.Series) -> Tuple[pd.Series, str]:
    """Return cleaned Series and a coarse dtype tag."""
    s = s.replace(r'^\s*$', np.nan, regex=True)        # blank → NaN
    num = pd.to_numeric(s, errors='coerce')
    if num.notna().sum() >= 0.8 * len(s.dropna()):
        return num.astype(float), "numeric"
    dt = pd.to_datetime(s, errors='coerce')
    if dt.notna().sum() >= 0.8 * len(s.dropna()):
        return dt.dt.normalize(), "datetime"
    return (
        s.astype(str).str.strip().str.casefold()
         .replace({"nan": np.nan, "none": np.nan}),
        "string",
    )

def _reason(tag: str, v1: Any, v2: Any, atol: float) -> str:
    """Classify why two values are not equal (after canonicalisation)."""
    if (pd.isna(v1) and not pd.isna(v2)) or (pd.isna(v2) and not pd.isna(v1)):
        return "missing-vs-value"
    if tag == "numeric":
        return f"numeric-delta>{atol}"
    if tag == "datetime":
        return "datetime-mismatch"
    return "string-diff"

# ─────────────────────────────────────────────────────────
# 1.  Main comparator
# ─────────────────────────────────────────────────────────
def compare_csv_to_excel_tab(csv_path: str,
                             xlsx_path: str,
                             target_tab: str,
                             numeric_atol: float = 1e-9,
                             verbose: bool = True
                            ) -> Tuple[bool, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Compare a CSV extract to a specific Excel sheet and return a rich report.

    Returns
    -------
    overall_ok : bool
        True iff *all* shared columns match row-by-row.
    summary_df : pd.DataFrame
        One row per shared column with pass/fail & mismatch count.
    detail_dict : Dict[str, pd.DataFrame]
        For every failing column, a dataframe listing each mismatching row
        with `csv`, `excel`, and `reason`.
    """
    # 1. Load
    df_csv   = pd.read_csv(csv_path)
    df_excel = pd.read_excel(xlsx_path, sheet_name=target_tab)

    # 2. Harmonise columns
    df_csv.columns   = df_csv.columns.str.strip().str.casefold()
    df_excel.columns = df_excel.columns.str.strip().str.casefold()
    shared_cols = df_csv.columns.intersection(df_excel.columns)
    if shared_cols.empty:
        raise ValueError("No shared columns to compare.")

    # 3. Align length (or align on a key of your choice beforehand)
    n = min(len(df_csv), len(df_excel))
    df_csv, df_excel = df_csv.iloc[:n].reset_index(drop=True), df_excel.iloc[:n].reset_index(drop=True)

    # 4. Loop columns
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
        passed     = mismatches == 0
        summary_rows.append(dict(column=col, dtype=tag, passed=passed, mismatches=mismatches))

        if not passed:
            overall_ok = False
            mismatch_rows = []
            for idx in np.flatnonzero(~equal_mask):
                v1, v2 = s1.iloc[idx], s2.iloc[idx]
                mismatch_rows.append(dict(row=idx, csv=v1, excel=v2,
                                          reason=_reason(tag, v1, v2, numeric_atol)))
            detail_dict[col] = pd.DataFrame(mismatch_rows)

    summary_df = pd.DataFrame(summary_rows)

    # 5. Pretty console output (optional)
    if verbose:
        print("\n=== Column-by-column summary ===")
        display(summary_df)

        for col, df in detail_dict.items():
            print(f"\n❌  Detail for column '{col}' ({len(df)} mismatches)\n")
            display(df)

        if overall_ok:
            print("\n✅  All shared columns match after normalisation.")

    return overall_ok, summary_df, detail_dict

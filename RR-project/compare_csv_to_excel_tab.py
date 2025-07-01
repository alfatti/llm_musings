import pandas as pd
import numpy as np
from typing import Tuple

def _canonicalise(s: pd.Series) -> Tuple[pd.Series, str]:
    """
    Try numeric ➜ datetime ➜ string conversions, returning the
    canonicalised series and a simple dtype tag.
    """
    # Treat excel empty strings as NaN
    s = s.replace(r'^\s*$', np.nan, regex=True)

    # 1️⃣ numeric?
    num = pd.to_numeric(s, errors='coerce')
    if num.notna().sum() >= 0.8 * len(s.dropna()):   # heuristic
        return num.astype(float), "numeric"

    # 2️⃣ datetime?
    dt = pd.to_datetime(s, errors='coerce', dayfirst=False)
    if dt.notna().sum() >= 0.8 * len(s.dropna()):
        return dt.dt.normalize(), "datetime"

    # 3️⃣ fallback → cleaned strings
    return (
        s.astype(str)
         .str.strip()
         .str.casefold()
         .replace({"nan": np.nan, "none": np.nan}),
        "string"
    )

def compare_csv_to_excel_tab(csv_path: str,
                             xlsx_path: str,
                             target_tab: str,
                             numeric_atol: float = 1e-9,
                             verbose: bool = True) -> bool:
    """
    Returns True if *every* shared column matches row-by-row
    (after type-robust canonicalisation); else False.
    """
    # 0. Load
    df_csv   = pd.read_csv(csv_path)
    df_excel = pd.read_excel(xlsx_path, sheet_name=target_tab)

    # 1. Harmonise column names
    df_csv.columns   = df_csv.columns.str.strip().str.casefold()
    df_excel.columns = df_excel.columns.str.strip().str.casefold()
    shared_cols = df_csv.columns.intersection(df_excel.columns)
    if shared_cols.empty:
        raise ValueError("No shared columns to compare!")

    # 2. Trim to common length (or decide on your own alignment key)
    n = min(len(df_csv), len(df_excel))
    df_csv   = df_csv.iloc[:n].reset_index(drop=True)
    df_excel = df_excel.iloc[:n].reset_index(drop=True)

    overall_ok = True
    for col in shared_cols:
        s1, tag1 = _canonicalise(df_csv[col])
        s2, tag2 = _canonicalise(df_excel[col])

        # Use the broader of the two tags
        tag = tag1 if tag1 == tag2 else "mixed"

        # 3. Column-specific comparison rule
        if tag == "numeric":
            equal = np.isclose(s1, s2, atol=numeric_atol, equal_nan=True)
        elif tag == "datetime":
            equal = s1.eq(s2) | (s1.isna() & s2.isna())
        else:  # string or mixed
            equal = s1.eq(s2) | (s1.isna() & s2.isna())

        if not equal.all():
            overall_ok = False
            if verbose:
                print(f"\n❌ Column '{col}' mismatches ({tag}):")
                mismatch_df = (pd.concat(
                    {"csv": s1, "excel": s2}, axis=1)
                    .loc[~equal]
                    .reset_index(names="row"))
                display(mismatch_df)  # Jupyter/IPython convenience

    if overall_ok and verbose:
        print("✅ All shared columns match after normalisation.")
    return overall_ok

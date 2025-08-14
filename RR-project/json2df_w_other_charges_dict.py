import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any, Tuple

def _safe_col(name: str) -> str:
    s = re.sub(r"[^0-9A-Za-z]+", "_", str(name)).strip("_")
    s = re.sub(r"_+", "_", s)
    return s.upper() or "OTHER"

def _normalize_charges(charges: Any, prefix: str) -> Dict[str, Any]:
    """
    Normalize a dict-like 'charges' object into flat columns with a prefix.
    - Non-dict inputs are treated as empty.
    - Keys are normalized to UPPER_SNAKE and prefixed (e.g., CHG_RENT).
    - If multiple original keys normalize to the same column, they are summed numerically.
    """
    if not isinstance(charges, dict):
        return {}
    flat = {}
    for k, v in charges.items():
        col = prefix + _safe_col(k)
        if col in flat:
            # Merge collisions by numeric sum (fallback: keep first)
            a = pd.to_numeric(flat[col], errors="coerce")
            b = pd.to_numeric(v, errors="coerce")
            if pd.isna(a) and pd.isna(b):
                pass  # keep existing as-is
            elif pd.isna(a):
                flat[col] = b
            elif pd.isna(b):
                flat[col] = a
            else:
                flat[col] = a + b
        else:
            flat[col] = v
    return flat

def records_to_df_with_errors(
    records: List[Dict[str, Any]],
    charges_key: str = "other_charges",
    prefix: str = "CHG_",
    numeric_cols_hint: List[str] = None,
    coerce_all_charges_numeric: bool = True,
    verbose: bool = True
) -> Tuple[pd.DataFrame, List[int]]:
    """
    Convert records (list of dicts) into a DataFrame, expanding dynamic charges to columns.
    Skips problematic records, prints their indices, and returns (df, error_page_num).

    Args
    ----
    records : list of dicts (each a unit row)
    charges_key : name of the nested dict with variable keys (default: 'other_charges')
    prefix : column prefix for expanded charges (default: 'CHG_')
    numeric_cols_hint : top-level fields you’d like coerced to numeric if present
    coerce_all_charges_numeric : coerce all CHG_* columns to numeric (default: True)
    verbose : print warnings for skipped rows

    Returns
    -------
    (df, error_page_num)
    """
    if numeric_cols_hint is None:
        numeric_cols_hint = ["sqft", "bedrooms", "bathrooms", "market_rent", "scheduled_rent", "balance"]

    rows = []
    error_page_num = []

    for idx, rec in enumerate(records):
        try:
            if not isinstance(rec, dict):
                raise TypeError(f"Record at idx {idx} is not a dict (got {type(rec).__name__}).")

            # Shallow copy to avoid mutating input
            row = dict(rec)

            # Pull and expand charges
            charges = row.pop(charges_key, {})
            flat_charges = _normalize_charges(charges, prefix=prefix)

            # Merge base fields and charges
            row.update(flat_charges)

            # Optional numeric coercion for hinted base fields (if present)
            for c in numeric_cols_hint:
                if c in row:
                    try:
                        row[c] = pd.to_numeric(row[c], errors="coerce")
                    except Exception:
                        pass

            rows.append(row)

        except Exception as exc:
            error_page_num.append(idx)
            if verbose:
                # Show a compact preview to help debugging without overwhelming output
                preview = str(rec)
                if len(preview) > 180:
                    preview = preview[:180] + "…"
                print(f"[WARN] Skipping record idx={idx}: {type(exc).__name__}: {exc}\n        Preview: {preview}")

    # Build DF from successful rows (may be empty)
    df = pd.DataFrame(rows) if rows else pd.DataFrame()

    # Coerce all charge columns to numeric if requested
    if coerce_all_charges_numeric and not df.empty:
        charge_cols = [c for c in df.columns if c.startswith(prefix)]
        for c in charge_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # Optional: put charge columns at the end for readability
        base_cols = [c for c in df.columns if c not in charge_cols]
        df = df[base_cols + sorted(charge_cols)]

    return df, error_page_num
#############################################
# records = [...]  # your clean list[dict], each with an 'other_charges' dict (variable keys)
df, error_page_num = records_to_df_with_errors(
    records,
    charges_key="other_charges",
    prefix="CHG_",
    numeric_cols_hint=["sqft","bedrooms","bathrooms","market_rent","scheduled_rent","balance"],
    coerce_all_charges_numeric=True,
    verbose=True
)

# Inspect
print("Errored indices:", error_page_num)
df.head()

import json, re
import pandas as pd
import numpy as np

def _extract_json_array(text: str) -> list:
    """
    Pulls the first top-level JSON array from arbitrary LLM text (handles code fences / preambles).
    Raises ValueError if no JSON array is found.
    """
    # Strip common code-fence wrappers
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.IGNORECASE)

    # Fast path: try direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass

    # Robust path: find the first [...] block with balanced brackets
    # (coarse but effective for typical LLM outputs)
    start_idxs = [m.start() for m in re.finditer(r"\[", text)]
    for s in start_idxs:
        depth = 0
        for i, ch in enumerate(text[s:], start=s):
            if ch == '[':
                depth += 1
            elif ch == ']':
                depth -= 1
                if depth == 0:
                    candidate = text[s:i+1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, list):
                            return obj
                    except Exception:
                        break
    raise ValueError("No valid top-level JSON array found in text.")

def _snake_upper(name: str) -> str:
    """Normalize a charge key into a safe COLUMN name (UPPER_SNAKE)."""
    # Replace non-alnum with underscores, collapse repeats, strip edges, upper-case
    s = re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_")
    s = re.sub(r"_+", "_", s)
    return s.upper() or "OTHER"

def llm_rentroll_to_df(llm_text: str, charge_prefix: str = "CHG_") -> pd.DataFrame:
    """
    Convert LLM JSON (array of unit dicts) into a flat DataFrame.
    Expands other_charges' dynamic keys into columns like CHG_RENT, CHG_GARBAGE, ...
    """
    records = _extract_json_array(llm_text)

    # Base frame from top-level fields
    df = pd.DataFrame.from_records(records)

    # Ensure other_charges exists and is a dict per row
    if "other_charges" not in df.columns:
        df["other_charges"] = [{}] * len(df)
    df["other_charges"] = df["other_charges"].apply(lambda d: d if isinstance(d, dict) else {})

    # Normalize charge keys per row, then expand
    def norm_charge_dict(d):
        out = {}
        for k, v in d.items():
            col = f"{charge_prefix}{_snake_upper(str(k))}"
            out[col] = v
        return out

    charges_expanded = df["other_charges"].apply(norm_charge_dict)
    charges_df = pd.json_normalize(charges_expanded).replace({None: np.nan})

    # Join and drop the nested dict column if you don’t need it
    df = pd.concat([df.drop(columns=["other_charges"]), charges_df], axis=1)

    # Optional: enforce numeric types on known numeric fields (ignore if missing)
    numeric_cols = [
        "sqft", "bedrooms", "bathrooms", "market_rent", "scheduled_rent", "balance"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Also coerce all CHG_* to numeric
    for c in df.columns:
        if c.startswith(charge_prefix):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Optional: sort columns (group charges to the end)
    base_cols = [c for c in df.columns if not c.startswith(charge_prefix)]
    charge_cols = sorted([c for c in df.columns if c.startswith(charge_prefix)])
    df = df[base_cols + charge_cols]

    return df

# -------------------------
# Example usage:
# llm_text = <your LLM raw string response here>
# df = llm_rentroll_to_df(llm_text, charge_prefix="CHG_")
# display(df.head())

import pandas as pd

# --------------------------------------------------------------------------- #
# CONFIGURATION
# --------------------------------------------------------------------------- #
LEASE_KEY_MAP = {
    "lease_type": "LeaseType",
    "lease_from": "LCD",
    "lease_to"  : "LXD",
    "term"      : "Term",
    "tenancy"   : "AvgTenantTenure",
    "area"      : "NRA"
}

RENT_STEPS_COLS = [
    "Charge", "Type", "Unit", "Area Label", "Area", "From", "To", "Monthly Amt",
    "Amt/Area", "Annual", "Annual/Area", "Management Fee", "Annual Gross Amount"
]

# --------------------------------------------------------------------------- #
# HELPERS
# --------------------------------------------------------------------------- #
def _as_rows(obj):
    """
    Normalise *obj* (which might be a DataFrame, list-of-lists, tuple-of-lists, etc.)
    into a plain Python list of rows. Guarantees that the result is iterable
    without triggering the 'truth value of a DataFrame is ambiguous' error.
    """
    if obj is None:
        return []
    if isinstance(obj, pd.DataFrame):
        return obj.values.tolist()
    if isinstance(obj, (list, tuple)):
        return list(obj)
    # Fallback for any other unexpected type
    return []

def _rent_metrics_for_unit(rent_steps_rows, unit):
    """
    Extract NRA, MonthlyRent, AnnualRent, $/sf for *unit* from rent-steps rows,
    and compute RentStep (latest annual − previous annual).
    """
    if not rent_steps_rows:          # no rent-step data at all
        return {"NRA": None, "MonthlyRent": None, "AnnualRent": None,
                "$/sf": None, "RentStep": None}

    df = pd.DataFrame(rent_steps_rows, columns=RENT_STEPS_COLS)
    df = df[df["Unit"].astype(str).str.strip() == unit]
    if df.empty:                     # unit not present in rent-steps
        return {"NRA": None, "MonthlyRent": None, "AnnualRent": None,
                "$/sf": None, "RentStep": None}

    df["From"] = pd.to_datetime(df["From"])
    df = df.sort_values("From")      # chronological

    latest = df.iloc[-1]
    prev   = df.iloc[-2] if len(df) >= 2 else None

    return {
        "NRA"        : latest["Area"],
        "MonthlyRent": latest["Monthly Amt"],
        "AnnualRent" : latest["Annual"],
        "$/sf"       : latest["Amt/Area"],
        "RentStep"   : (latest["Annual"] - prev["Annual"]) if prev is not None else None
    }

# --------------------------------------------------------------------------- #
# MAIN ROUTINE
# --------------------------------------------------------------------------- #
def unpivot_rent_roll(rent_roll: dict) -> pd.DataFrame:
    """
    Flatten the nested `rent_roll` structure so every (property, tenant, unit)
    becomes one row, with lease fields renamed and rent-step metrics appended.
    """
    rows = []

    for property_name, prop_data in rent_roll.items():
        building_id = prop_data.get("buildingID", "")
        tenants     = prop_data.get("tenants", {})

        for tenant_name, tenant_info in tenants.items():
            tenant_id   = tenant_info.get("tenant_id", "")
            unit_range  = tenant_info.get("unit_range", "")
            lease_info  = tenant_info.get("lease", {}) or {}

            # NEW: robust, ambiguity-free extraction of rent_steps
            rent_steps_rows = _as_rows(tenant_info.get("rent_steps"))

            # Rename lease keys on the fly
            lease_renamed = {LEASE_KEY_MAP.get(k, k): v for k, v in lease_info.items()}

            # Fan-out one row per unit
            units = [u.strip() for u in unit_range.split(",") if u.strip()]
            for unit in units:
                rent_metrics = _rent_metrics_for_unit(rent_steps_rows, unit)

                rows.append({
                    "property_name": property_name,
                    "building_id"  : building_id,
                    "tenant_name"  : tenant_name,
                    "tenant_id"    : tenant_id,
                    "unit"         : unit,
                    **lease_renamed,
                    **rent_metrics
                })

    return pd.DataFrame(rows)

# --------------------------------------------------------------------------- #
# EXAMPLE USAGE
# --------------------------------------------------------------------------- #
# df = unpivot_rent_roll(rent_roll)
# display(df.head())

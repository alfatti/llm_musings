import pandas as pd
from datetime import datetime

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
    """Normalise *obj* (which might be a DataFrame, list-of-lists, tuple-of-lists, etc.) into a list of rows."""
    if obj is None:
        return []
    if isinstance(obj, pd.DataFrame):
        return obj.values.tolist()
    if isinstance(obj, (list, tuple)):
        return list(obj)
    return []

def _rent_metrics_for_unit(rent_steps_rows, unit):
    """Compute rent metrics for the given unit including forward-looking RentStep and EffectiveRent."""
    if not rent_steps_rows:
        return {"NRA": None, "MonthlyRent": None, "AnnualRent": None,
                "$/sf": None, "RentStep": None, "EffectiveRent": None}

    df = pd.DataFrame(rent_steps_rows, columns=RENT_STEPS_COLS)
    df = df[df["Unit"].astype(str).str.strip() == unit]
    if df.empty:
        return {"NRA": None, "MonthlyRent": None, "AnnualRent": None,
                "$/sf": None, "RentStep": None, "EffectiveRent": None}

    df["From"] = pd.to_datetime(df["From"])
    df["To"]   = pd.to_datetime(df["To"])
    df = df.sort_values("From")

    year_now = pd.Timestamp.now().year

    # Find "current period" and "next period" based on forward-looking logic
    current_row = df[df["To"].dt.year == year_now]
    next_row    = df[(df["From"].dt.year == year_now) & (df["To"].dt.year == year_now + 1)]

    current = current_row.iloc[0] if not current_row.empty else None
    next_   = next_row.iloc[0] if not next_row.empty else None

    rentstep = (next_["Annual"] - current["Annual"]) if (current is not None and next_ is not None) else None
    effectiverent = next_["Annual"] if next_ is not None else None

    latest = df.iloc[-1]  # still use latest for snapshot metrics

    return {
        "NRA"           : latest["Area"],
        "MonthlyRent"   : latest["Monthly Amt"],
        "AnnualRent"    : latest["Annual"],
        "$/sf"          : latest["Amt/Area"],
        "RentStep"      : rentstep,
        "EffectiveRent" : effectiverent
    }

# --------------------------------------------------------------------------- #
# MAIN FUNCTION
# --------------------------------------------------------------------------- #
def unpivot_rent_roll(rent_roll: dict) -> pd.DataFrame:
    """Flatten the nested `rent_roll` structure and compute rent metrics."""
    rows = []

    for property_name, prop_data in rent_roll.items():
        building_id = prop_data.get("buildingID", "")
        tenants     = prop_data.get("tenants", {})

        for tenant_name, tenant_info in tenants.items():
            tenant_id   = tenant_info.get("tenant_id", "")
            unit_range  = tenant_info.get("unit_range", "")
            lease_info  = tenant_info.get("lease", {}) or {}

            rent_steps_rows = _as_rows(tenant_info.get("rent_steps"))

            # Rename lease keys
            lease_renamed = {LEASE_KEY_MAP.get(k, k): v for k, v in lease_info.items()}

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

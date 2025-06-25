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
    if obj is None:
        return []
    if isinstance(obj, pd.DataFrame):
        return obj.values.tolist()
    if isinstance(obj, (list, tuple)):
        return list(obj)
    return []

def _rent_metrics_for_unit(rent_steps_rows, unit):
    """
    Extract forward-looking rent step for the given unit:
    - Current period: From.year == this year, To.year == this year + 1
    - Next period   : From.year == this year + 1, To.year == this year + 2
    """
    if not rent_steps_rows:
        return {
            "NRA": None, "MonthlyRent": None, "AnnualRent": None,
            "$/sf": None, "RentStep": None, "EffectiveRent": None
        }

    df = pd.DataFrame(rent_steps_rows, columns=RENT_STEPS_COLS)
    df = df[df["Unit"].astype(str).str.strip() == unit]
    if df.empty:
        return {
            "NRA": None, "MonthlyRent": None, "AnnualRent": None,
            "$/sf": None, "RentStep": None, "EffectiveRent": None
        }

    df["From"] = pd.to_datetime(df["From"], errors="coerce")
    df["To"]   = pd.to_datetime(df["To"], errors="coerce")

    year_now = datetime.now().year

    current = df[
        (df["From"].dt.year == year_now) &
        (df["To"].dt.year == year_now + 1)
    ]

    next_ = df[
        (df["From"].dt.year == year_now + 1) &
        (df["To"].dt.year == year_now + 2)
    ]

    if current.empty:
        return {
            "NRA": None, "MonthlyRent": None, "AnnualRent": None,
            "$/sf": None, "RentStep": None, "EffectiveRent": None
        }

    # Use first match in case of duplicates
    current_row = current.iloc[0]
    next_row    = next.iloc[0] if not next_.empty else None

    return {
        "NRA": current_row["Area"],
        "MonthlyRent": current_row["Monthly Amt"],
        "AnnualRent" : current_row["Annual"],
        "$/sf"       : current_row["Amt/Area"],
        "RentStep"   : (next_row["Annual"] - current_row["Annual"]) if next_row is not None else None,
        "EffectiveRent": next_row["Annual"] if next_row is not None else None
    }

# --------------------------------------------------------------------------- #
# MAIN ROUTINE
# --------------------------------------------------------------------------- #
def unpivot_rent_roll(rent_roll: dict) -> pd.DataFrame:
    rows = []

    for property_name, prop_data in rent_roll.items():
        building_id = prop_data.get("buildingID", "")
        tenants     = prop_data.get("tenants", {})

        for tenant_name, tenant_info in tenants.items():
            tenant_id   = tenant_info.get("tenant_id", "")
            unit_range  = tenant_info.get("unit_range", "")
            lease_info  = tenant_info.get("lease", {}) or {}

            rent_steps_rows = _as_rows(tenant_info.get("rent_steps"))

            lease_renamed = {
                LEASE_KEY_MAP.get(k, k): v
                for k, v in lease_info.items()
            }

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
# Example:
# df = unpivot_rent_roll(rent_roll)
# display(df)

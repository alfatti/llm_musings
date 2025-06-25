import pandas as pd
from datetime import datetime

# ---- configuration ----------------------------------------------------------
LEASE_KEY_MAP = {
    "lease_type": "LeaseType",
    "lease_from": "LCD",
    "lease_to"  : "LXD",
    "term"      : "Term",
    "tenancy"   : "AvgTenantTenure",
    "area"      : "NRA"          # still kept in case it appears in `lease`
}

RENT_STEPS_COLS = [
    "Charge", "Type", "Unit", "Area Label", "Area", "From", "To", "Monthly Amt",
    "Amt/Area", "Annual", "Annual/Area", "Management Fee", "Annual Gross Amount"
]

# ---- helpers ----------------------------------------------------------------
def _rent_metrics_for_unit(rent_steps_rows, unit):
    """Return dict with NRA, MonthlyRent, AnnualRent, $/sf, RentStep for one unit."""
    if not rent_steps_rows:
        # No rent-step data at all
        return {"NRA": None, "MonthlyRent": None, "AnnualRent": None,
                "$/sf": None, "RentStep": None}

    df_rs = pd.DataFrame(rent_steps_rows, columns=RENT_STEPS_COLS)
    df_rs = df_rs[df_rs["Unit"].astype(str).str.strip() == unit]
    if df_rs.empty:
        # Unit not present in rent-steps
        return {"NRA": None, "MonthlyRent": None, "AnnualRent": None,
                "$/sf": None, "RentStep": None}

    # Make sure we can sort chronologically
    df_rs["From"] = pd.to_datetime(df_rs["From"])
    df_rs = df_rs.sort_values("From")             # oldest → newest

    latest = df_rs.iloc[-1]
    prev   = df_rs.iloc[-2] if len(df_rs) >= 2 else None

    return {
        "NRA"        : latest["Area"],
        "MonthlyRent": latest["Monthly Amt"],
        "AnnualRent" : latest["Annual"],
        "$/sf"       : latest["Amt/Area"],
        "RentStep"   : (latest["Annual"] - prev["Annual"]) if prev is not None else None
    }

# ---- main routine -----------------------------------------------------------
def unpivot_rent_roll(rent_roll):
    rows = []

    for property_name, prop_data in rent_roll.items():
        building_id = prop_data.get("buildingID", "")
        tenants     = prop_data.get("tenants", {})

        for tenant_name, tenant_info in tenants.items():
            tenant_id   = tenant_info.get("tenant_id", "")
            unit_range  = tenant_info.get("unit_range", "")
            lease_info  = tenant_info.get("lease",     {}) or {}
            rent_steps  = tenant_info.get("rent_steps", []) or []

            # 1) Rename lease keys on the fly
            lease_renamed = {LEASE_KEY_MAP.get(k, k): v for k, v in lease_info.items()}

            # 2) Fan-out one row per unit
            units = [u.strip() for u in unit_range.split(",") if u.strip()]
            for unit in units:
                rent_metrics = _rent_metrics_for_unit(rent_steps, unit)

                row = {
                    "property_name": property_name,
                    "building_id"  : building_id,
                    "tenant_name"  : tenant_name,
                    "tenant_id"    : tenant_id,
                    "unit"         : unit,
                    **lease_renamed,
                    **rent_metrics
                }
                rows.append(row)

    return pd.DataFrame(rows)

# ----------------------------------------------------------------------------- 
# Example usage:
# df = unpivot_rent_roll(rent_roll)
# print(df.head())

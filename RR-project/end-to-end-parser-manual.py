import pandas as pd
from collections import defaultdict
import re
from typing import Dict, Any

# ---------- helpers to build defaultdict hierarchy ----------
def _empty_tenant() -> Dict[str, Any]:
    return {
        "unit_range": None,
        "lease": {},
        "rent_steps": pd.DataFrame(),
        "charge_schedule": pd.DataFrame(),
        "amendment": {}
    }

def _tenants_dict() -> defaultdict:
    return defaultdict(_empty_tenant)

def _property_data() -> Dict[str, Any]:
    return {
        "buildingID": None,
        "tenants": _tenants_dict()
    }

# ---------- core processing function ----------
def process_rent_roll(file_path: str) -> defaultdict:
    \"\"\"Read *file_path* (Excel) and return a nested defaultdict structure.\"\"\"
    df = pd.read_excel(file_path)

    # Identify the start of each tenant block by non‑null 'Lease'
    tenant_starts = df[df[\"Lease\"].notna()].index.tolist()
    rent_roll: defaultdict = defaultdict(_property_data)

    for idx, start in enumerate(tenant_starts):
        end = tenant_starts[idx + 1] if idx + 1 < len(tenant_starts) else len(df)
        chunk = df.iloc[start:end].reset_index(drop=True)

        # ---------- extract property / tenant headers ----------
        property_name: str = chunk.loc[0, \"Property\"]
        tenant_name: str = chunk.loc[0, \"Lease\"]
        unit_range = chunk.loc[0, \"Unit(s)\"]
        building_match = re.search(r\"\\((.*?)\\)\", property_name or \"\")
        building_id = building_match.group(1) if building_match else None

        prop_entry = rent_roll[property_name]
        prop_entry[\"buildingID\"] = building_id
        tenant_entry = prop_entry[\"tenants\"][tenant_name]
        tenant_entry[\"unit_range\"] = unit_range
        tenant_entry[\"lease\"] = {
            \"lease_type\": chunk.loc[0, \"Lease Type\"],
            \"lease_from\": chunk.loc[0, \"Lease From\"],
            \"lease_to\": chunk.loc[0, \"Lease To\"],
            \"term\": chunk.loc[0, \"Term\"],
            \"tenancy\": chunk.loc[0, \"Tenancy\"],
            \"monthly\": chunk.loc[0, \"Monthly\"],
            \"annual\": chunk.loc[0, \"Annual\"],
            \"area\": chunk.loc[0, \"Area\"],
        }

        # ---------- split body (rows after metadata) by blank lines ----------
        body = chunk.iloc[1:].reset_index(drop=True)
        blank_rows = body[body.isnull().all(axis=1)].index.tolist()

        sections = []
        prev = 0
        for br in blank_rows + [len(body)]:
            section = body.iloc[prev:br].dropna(how=\"all\").reset_index(drop=True)
            if not section.empty:
                sections.append(section)
            prev = br + 1  # skip the blank row itself

        if len(sections) >= 1:
            tenant_entry[\"rent_steps\"] = sections[0]
        if len(sections) >= 2:
            tenant_entry[\"charge_schedule\"] = sections[1]
        if len(sections) >= 3:
            amend_row = sections[2].iloc[0]
            tenant_entry[\"amendment\"] = {
                \"type\": amend_row.get(\"Amendment\"),
                \"status\": amend_row.get(\"Type.1\"),
                \"from\": amend_row.get(\"Status\"),
                \"to\": amend_row.get(\"Move In\"),
                \"area\": amend_row.get(\"Area\"),
                \"description\": amend_row.get(\"Description\"),
                \"notes\": \"\",
            }

    return rent_roll

# ---------- CLI helper ----------
if __name__ == \"__main__\":  # pragma: no cover
    import argparse, pprint
    ap = argparse.ArgumentParser(description=\"Process a rent‑roll Excel file.\")
    ap.add_argument(\"file\", help=\"Path to .xlsx rent‑roll

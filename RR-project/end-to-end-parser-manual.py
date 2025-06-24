
import pandas as pd
from collections import defaultdict
import re

# Initialization factories
def empty_tenant():
    return {
        "unit_range": None,
        "lease": {},
        "rent_steps": pd.DataFrame(),
        "charge_schedule": pd.DataFrame(),
        "amendment": {}
    }

def tenants_dict():
    return defaultdict(empty_tenant)

def property_data():
    return {
        "buildingID": None,
        "tenants": tenants_dict()
    }

def process_rent_roll(file_path):
    df = pd.read_excel(file_path)

    rent_roll = defaultdict(property_data)

    tenant_starts = df[df["Lease"].notna()].index.tolist()

    for i, start in enumerate(tenant_starts):
        end = tenant_starts[i + 1] if i + 1 < len(tenant_starts) else len(df)
        chunk = df.iloc[start:end].reset_index(drop=True)

        property_name = chunk.loc[0, "Property"]
        building_id_match = re.search(r"\\((.*?)\\)", property_name or "")
        building_id = building_id_match.group(1) if building_id_match else None

        tenant_name = chunk.loc[0, "Lease"]
        rent_roll[property_name]["buildingID"] = building_id
        tenant = rent_roll[property_name]["tenants"][tenant_name]

        tenant["unit_range"] = chunk.loc[0, "Unit(s)"]
        tenant["lease"] = {
            "lease_type": chunk.loc[0, "Lease Type"],
            "lease_from": chunk.loc[0, "Lease From"],
            "lease_to": chunk.loc[0, "Lease To"],
            "term": chunk.loc[0, "Term"],
            "tenancy": chunk.loc[0, "Tenancy"],
            "monthly": chunk.loc[0, "Monthly"],
            "annual": chunk.loc[0, "Annual"],
            "area": chunk.loc[0, "Area"]
        }

        rent_df = chunk[chunk["Charge"] == "rnt"]
        charge_df = chunk[(chunk["Type"] == "CAM") | (chunk["Charge"].isin(["capcam", "capimp", "elec", "ins", "mng", "ope", "pbx"]))]

        tenant["rent_steps"] = rent_df[[
            "Charge", "Type", "Unit", "Area Label", "From", "To", "Monthly Amt", "Amt/Area", "Annual"
        ]].dropna(how="all").reset_index(drop=True)

        tenant["charge_schedule"] = charge_df[[
            "Charge", "Type", "Unit", "Area Label", "From", "To", "Monthly Amt", "Amt/Area", "Annual"
        ]].dropna(how="all").reset_index(drop=True)

        if pd.notna(chunk.loc[0, "Amendment"]):
            tenant["amendment"] = {
                "type": chunk.loc[0, "Amendment"],
                "status": chunk.loc[0, "Type.1"],
                "from": chunk.loc[0, "Status"],
                "to": chunk.loc[0, "Move In"],
                "area": chunk.loc[0, "Area"],
                "description": chunk.loc[0, "Description"],
                "notes": ""
            }

    return rent_roll

# Example usage:
# rent_roll = process_rent_roll("path_to_your_excel_file.xlsx")
# print(rent_roll)
"""

with open("/mnt/data/process_rent_roll.py", "w") as f:
    f.write(script_content)

"/mnt/data/process_rent_roll.py"

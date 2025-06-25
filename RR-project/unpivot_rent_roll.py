import pandas as pd

def unpivot_rent_roll(rent_roll):
    records = []

    for property_name, property_data in rent_roll.items():
        building_id = property_data.get("buildingID", "")
        tenants = property_data.get("tenants", {})

        for tenant_name, tenant_info in tenants.items():
            unit_range = tenant_info.get("unit_range", "")
            lease_info = tenant_info.get("lease", {})
            
            # Attempt to extract tenant ID from tenant name if it’s in form "Name (ID)"
            if "(" in tenant_name and tenant_name.endswith(")"):
                name_parts = tenant_name.rsplit("(", 1)
                name_clean = name_parts[0].strip()
                tenant_id = name_parts[1].rstrip(")")
            else:
                name_clean = tenant_name
                tenant_id = ""

            units = [u.strip() for u in unit_range.split(",") if u.strip()]
            for unit in units:
                record = {
                    "property_name": property_name,
                    "building_id": building_id,
                    "tenant_name": name_clean,
                    "tenant_id": tenant_id,
                    "unit": unit,
                    **lease_info
                }
                records.append(record)

    return pd.DataFrame(records)

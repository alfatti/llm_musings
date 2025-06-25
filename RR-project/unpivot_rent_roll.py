import pandas as pd

def unpivot_rent_roll(rent_roll):
    rows = []

    for property_name, prop_data in rent_roll.items():
        building_id = prop_data.get("buildingID")
        tenants = prop_data.get("tenants", {})

        for tenant_name, tenant_data in tenants.items():
            lease = tenant_data.get("lease", {})
            unit_range = tenant_data.get("unit_range", "")
            units = [u.strip() for u in unit_range.split(",") if u.strip()]

            tenant_id = lease.get("tenantID", None)  # optional

            for unit in units:
                row = {
                    "property_name": property_name,
                    "buildingID": building_id,
                    "tenant_name": tenant_name,
                    "tenantID": tenant_id,
                    "unit": unit,
                }
                # Add all lease parameters to the row
                row.update(lease)
                rows.append(row)

    return pd.DataFrame(rows)

# Example usage
# df = unpivot_rent_roll(rent_roll)

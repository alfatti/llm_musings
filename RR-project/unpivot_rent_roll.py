import pandas as pd

def unpivot_rent_roll(rent_roll):
    records = []

    lease_key_map = {
        'lease_type': 'LeaseType',
        'lease_from': 'LCD',
        'lease_to': 'LXD',
        'term': 'Term',
        'tenancy': 'AvgTenantTenure',
        'area': 'NRA'
    }

    for property_name, property_data in rent_roll.items():
        building_id = property_data.get("buildingID", "")
        tenants = property_data.get("tenants", {})

        for tenant_name, tenant_info in tenants.items():
            unit_range = tenant_info.get("unit_range", "")
            lease_info = tenant_info.get("lease", {})
            tenant_id = tenant_info.get("tenant_id", "")

            # Rename lease keys
            renamed_lease_info = {
                lease_key_map.get(k, k): v
                for k, v in lease_info.items()
            }

            units = [u.strip() for u in unit_range.split(",") if u.strip()]
            for unit in units:
                record = {
                    "property_name": property_name,
                    "building_id": building_id,
                    "tenant_name": tenant_name,
                    "tenant_id": tenant_id,
                    "unit": unit,
                    **renamed_lease_info
                }
                records.append(record)

    return pd.DataFrame(records)

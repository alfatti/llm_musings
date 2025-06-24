from collections import defaultdict

def tenant_data():
    return {
        "leases": [],
        "charges": [],
        # Add other fields as needed
    }

def tenants_dict():
    return defaultdict(tenant_data)

def property_data():
    return {
        "buildingID": None,
        "tenants": tenants_dict()
    }

properties = defaultdict(property_data)

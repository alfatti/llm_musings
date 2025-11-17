import re

def split_case_id_and_category(s: str):
    """
    Splits a string of the form '12345- CATEGORY' into (case_id, category).
    Handles extra spaces and multiple hyphens in the category.
    
    Returns:
        (case_id, category) both as strings, or (None, None) if input invalid.
    """
    if not isinstance(s, str) or "-" not in s:
        return None, None
    
    # Split on the FIRST hyphen only
    parts = s.split("-", 1)
    raw_id = parts[0].strip()
    raw_cat = parts[1].strip()
    
    # Clean up: ensure ID is digits-only
    case_id = re.sub(r"[^\d]", "", raw_id) if raw_id else None
    category = raw_cat if raw_cat else None
    
    return case_id, category
##############################
df["CaseID"], df["Category"] = zip(*df["CombinedColumn"].map(split_case_id_and_category))

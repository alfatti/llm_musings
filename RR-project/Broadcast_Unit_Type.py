import pandas as pd
import re

# Load Excel file
df = pd.read_excel("rent_roll.xlsx", header=0)

# Initialize
current_unit_type = None
last_main_row = None
rows = []

for _, row in df.iterrows():
    first_cell = str(row.iloc[0]) if pd.notna(row.iloc[0]) else ""

    # --- Detect "Unit Type: ..." ---
    if first_cell.startswith("Unit Type:"):
        match = re.match(r"Unit Type:\s*(.*)", first_cell)
        if match:
            current_unit_type = match.group(1).strip()
        continue

    # --- Skip empty rows ---
    if row.notna().sum() < 2:
        continue

    # --- Determine if it's a new main unit row (non-empty 'Unit') ---
    is_new_unit = pd.notna(row.get("Unit"))

    # If it's a new unit, update tracker
    if is_new_unit:
        last_main_row = row.to_dict()
        last_main_row["Unit Type"] = current_unit_type
        rows.append(last_main_row)
    else:
        # For charge sub-rows, forward-fill main row info
        if last_main_row:
            filled_row = row.copy()
            for col in ["Unit", "Address", "SQFT", "Status"]:
                filled_row[col] = last_main_row.get(col)
            filled_row["Unit Type"] = current_unit_type
            rows.append(filled_row.to_dict())

    # Check if next row is a different unit or new section — insert separator
    next_index = _ + 1
    if next_index < len(df):
        next_row = df.iloc[next_index]
        next_first_cell = str(next_row.iloc[0]) if pd.notna(next_row.iloc[0]) else ""
        if next_first_cell.startswith("Unit Type:") or pd.notna(next_row.get("Unit")):
            rows.append({})  # empty separator row

# Create DataFrame
final_df = pd.DataFrame(rows)

# Save or show
print(final_df)
# final_df.to_excel("unpivoted_rent_roll_with_blocks.xlsx", index=False)

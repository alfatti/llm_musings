import pandas as pd
import re

# Load your Excel file (adjust as needed)
df = pd.read_excel("rent_roll.xlsx", header=0)

# Columns you want to forward-fill when blank in sub-rows
forward_cols = ["Unit", "Address", "SQFT", "Status"]

# Initialize
current_unit_type = None
last_main_row = None
rows = []

for idx, row in df.iterrows():
    first_cell = str(row.iloc[0]) if pd.notna(row.iloc[0]) else ""

    # --- Detect "Unit Type: ..." marker ---
    if first_cell.startswith("Unit Type:"):
        match = re.match(r"Unit Type:\s*(.*)", first_cell)
        if match:
            current_unit_type = match.group(1).strip()
            print(f"[{idx}] Found Unit Type: {current_unit_type}")
        continue

    # --- Skip mostly empty rows ---
    if row.notna().sum() < 2:
        continue

    # --- Determine if it's a new unit (based on non-empty 'Unit') ---
    is_new_unit = pd.notna(row.get("Unit"))

    # If new main row, store as last_main_row
    if is_new_unit:
        last_main_row = row.to_dict()
        last_main_row["Unit Type"] = current_unit_type
        rows.append(last_main_row)
    else:
        # Charge sub-row: forward-fill unit data
        if last_main_row:
            filled_row = row.copy()
            for col in forward_cols:
                filled_row[col] = last_main_row.get(col)
            filled_row["Unit Type"] = current_unit_type
            rows.append(filled_row.to_dict())
        else:
            print(f"[{idx}] Warning: sub-row found before any main row — skipped")
            continue

    # --- Insert separator row if next row is a new unit or Unit Type ---
    next_index = idx + 1
    if next_index < len(df):
        next_row = df.iloc[next_index]
        next_first = str(next_row.iloc[0]) if pd.notna(next_row.iloc[0]) else ""
        is_unit_type = next_first.startswith("Unit Type:")
        is_new_unit = pd.notna(next_row.get("Unit"))

        if is_unit_type or is_new_unit:
            rows.append({})  # blank row separator

# Build final DataFrame
final_df = pd.DataFrame(rows)

# If still empty, show a sample of `rows`
if final_df.empty:
    print("Final DataFrame is empty. Sample of raw rows:")
    for r in rows[:5]:
        print(r)

# Preview
print(final_df.head(10))
# Optionally save
# final_df.to_excel("unpivoted_rent_roll_with_blocks.xlsx", index=False)

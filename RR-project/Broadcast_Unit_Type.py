import pandas as pd
import re

# Load Excel file
df = pd.read_excel("rent_roll.xlsx", header=0)

# Initialize variables
current_unit_type = None
records = []

for _, row in df.iterrows():
    first_cell = str(row.iloc[0]) if pd.notna(row.iloc[0]) else ""

    # Detect Unit Type marker like "Unit Type: Studio"
    if first_cell.startswith("Unit Type:"):
        # Extract the unit type using regex or string split
        match = re.match(r"Unit Type:\s*(.*)", first_cell)
        if match:
            current_unit_type = match.group(1).strip()
        continue  # Skip this row

    # Skip rows that are mostly empty
    if row.notna().sum() < 2:
        continue

    # Valid data row — attach the unit type
    row_data = row.to_dict()
    row_data["Unit Type"] = current_unit_type
    records.append(row_data)

# Assemble final DataFrame
unpivoted_df = pd.DataFrame(records)

# Optional cleanup
unpivoted_df = unpivoted_df.dropna(subset=["Unit"])  # if "Unit" is the key column

# Save or display
print(unpivoted_df)
# unpivoted_df.to_excel("unpivoted_rent_roll_unpivoted.xlsx", index=False)

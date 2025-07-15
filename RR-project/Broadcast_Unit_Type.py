import pandas as pd

# Load the sheet (adjust header row as needed)
df = pd.read_excel("rent_roll.xlsx", header=0)

# Initialize state
current_unit_type = None
rows = []

# Iterate over each row
for _, row in df.iterrows():
    first_cell = row.iloc[0]

    # Detect if row contains a Unit Type label (non-numeric + rest of row mostly empty)
    if isinstance(first_cell, str) and row[1:].isna().all():
        current_unit_type = first_cell
        continue

    # Normal data row
    if pd.notna(first_cell):
        row_data = row.to_dict()
        row_data["Unit Type"] = current_unit_type
        rows.append(row_data)

# Final unpivoted DataFrame
unpivoted_df = pd.DataFrame(rows)

# Drop rows that are still blank (just in case)
unpivoted_df = unpivoted_df.dropna(subset=["Unit"])

# Save or display
print(unpivoted_df)
# unpivoted_df.to_excel("unpivoted_rent_roll.xlsx", index=False)

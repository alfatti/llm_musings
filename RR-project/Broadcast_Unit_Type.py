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

# Insert empty row between unit chunks (based on NaNs in 'Unit' column)
break_indices = unpivoted_df.index[unpivoted_df["Unit"].notna() & unpivoted_df["Unit"].shift().isna()]
empty_rows = pd.DataFrame([{}] * len(break_indices), columns=unpivoted_df.columns)
unpivoted_df_with_gaps = pd.concat(
    [
        unpivoted_df.loc[:i - 1].append(empty_rows.iloc[[j]])
        if j < len(empty_rows)
        else unpivoted_df.loc[:i - 1]
        for j, i in enumerate(break_indices.tolist() + [len(unpivoted_df)])
    ],
    ignore_index=True
)

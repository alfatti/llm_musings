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

# Step 1: Find indices where a new unit starts after NaNs
break_indices = unpivoted_df.index[unpivoted_df["Unit"].notna() & unpivoted_df["Unit"].shift().isna()].tolist()

# Step 2: Prepare output chunks with blank rows in between
chunks = []
prev_idx = 0

for idx in break_indices + [len(unpivoted_df)]:
    chunk = unpivoted_df.iloc[prev_idx:idx]
    chunks.append(chunk)
    chunks.append(pd.DataFrame([{}], columns=unpivoted_df.columns))  # blank row
    prev_idx = idx

# Step 3: Combine
unpivoted_df_with_gaps = pd.concat(chunks, ignore_index=True)


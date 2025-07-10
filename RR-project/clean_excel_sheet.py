import pandas as pd

# --- Load the spreadsheet, without header ---
df_raw = pd.read_excel("your_file.xlsx", header=None)

# --- Step 1: Detect the start of the actual header ---
for i in range(len(df_raw)):
    if df_raw.iloc[i].notna().sum() > len(df_raw.columns) // 2:
        header_start = i
        break

# --- Step 2: Merge split headers ---
header_row_1 = df_raw.iloc[header_start].fillna("")
header_row_2 = df_raw.iloc[header_start + 1].fillna("")

final_headers = [
    f"{str(a).strip()}_{str(b).strip()}".strip("_")
    if a or b else f"col_{idx}"
    for idx, (a, b) in enumerate(zip(header_row_1, header_row_2))
]

# --- Step 3: Extract the data part ---
df_clean = df_raw.iloc[header_start + 2:].reset_index(drop=True)
df_clean.columns = final_headers

# --- Step 4: Remove non-tabular rows like 'Current Residents' ---
# Drop rows where number of non-null values is very small (e.g., 1 or less)
threshold = 2  # Can tune this if some columns are always empty
df_clean = df_clean[df_clean.notna().sum(axis=1) >= threshold].reset_index(drop=True)

# Optional: convert numeric columns where possible
df_clean = df_clean.apply(pd.to_numeric, errors='ignore')

# --- Done ---
print(df_clean.head())

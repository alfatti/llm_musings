import pandas as pd
import numpy as np

# Example dataframe
data = [
    ["Header", "Header"],
    ["A", 1],
    ["B", 2],
    [np.nan, np.nan],
    ["C", 3],
    ["D", 4],
    [np.nan, np.nan],
    ["E", 5],
]
df = pd.DataFrame(data, columns=["col1", "col2"])

# Start from the second row (drop first)
df_rest = df.iloc[1:].reset_index(drop=True)

# Find indices of all-NaN rows
nan_mask = df_rest.isna().all(axis=1)
nan_indices = nan_mask[nan_mask].index.tolist()

# Include 0 as the start and len(df_rest) as the end
boundaries = [0] + nan_indices + [len(df_rest)]

# Extract chunks
chunks = []
for i in range(len(boundaries) - 1):
    start = boundaries[i]
    end = boundaries[i + 1]
    chunk = df_rest.iloc[start:end].dropna(how="all").reset_index(drop=True)
    if not chunk.empty:
        chunks.append(chunk)

# ✅ `chunks` is your final list of sub-DataFrames with reset index
# Example usage: print them
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}:\n{chunk}")

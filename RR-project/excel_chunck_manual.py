import pandas as pd

# Load the Excel sheet
df = pd.read_excel("your_file.xlsx")  # Replace with your actual path

# Step 1: Drop fully empty rows (blank separators)
df_cleaned = df.dropna(how='all')

# Step 2: Identify the start indices of each chunk
chunk_starts = df_cleaned[
    df_cleaned['Property'].notna() &
    df_cleaned['Unit(s)'].notna() &
    df_cleaned['Lease'].notna()
].index.tolist()

# Step 3: Add the final index to close the last chunk
chunk_starts.append(df_cleaned.index[-1] + 1)

# Step 4: Create the list of chunked DataFrames
chunk_dfs = [
    df_cleaned.loc[chunk_starts[i]:chunk_starts[i+1]-1].copy()
    for i in range(len(chunk_starts)-1)
]

# Now `chunk_dfs` is a list of DataFrames, each representing one property chunk

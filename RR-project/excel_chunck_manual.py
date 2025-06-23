import pandas as pd

df = pd.read_excel("your_file.xlsx")

# Drop completely empty rows just in case
df_cleaned = df.dropna(how='all')

# Identify rows where a new chunk starts
chunk_starts = df_cleaned[
    df_cleaned['Property'].notna() & 
    df_cleaned['Unit(s)'].notna() & 
    df_cleaned['Lease'].notna()
].index.tolist()

# Add end of file as a final chunk end boundary
chunk_starts.append(df_cleaned.index[-1] + 1)

# Slice chunks
chunks = []
for i in range(len(chunk_starts) - 1):
    start_idx = chunk_starts[i]
    end_idx = chunk_starts[i + 1]
    chunk_df = df_cleaned.loc[start_idx:end_idx - 1]
    chunks.append(chunk_df)



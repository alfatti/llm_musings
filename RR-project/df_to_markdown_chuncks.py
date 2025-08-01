import pandas as pd
import math

def split_df_to_markdown_chunks(df: pd.DataFrame, rows_per_chunk: int = 30):
    """
    Splits a DataFrame into markdown-formatted chunks, each with headers.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to split.
        rows_per_chunk (int): Max number of data rows per chunk (headers not counted).
    
    Returns:
        List[str]: Markdown-formatted table strings.
    """
    num_chunks = math.ceil(len(df) / rows_per_chunk)
    markdown_chunks = []

    for i in range(num_chunks):
        chunk = df.iloc[i * rows_per_chunk : (i + 1) * rows_per_chunk]
        markdown = chunk.to_markdown(index=False)  # Include column headers, omit index
        markdown_chunks.append(markdown)

    return markdown_chunks

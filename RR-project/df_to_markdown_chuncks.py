import pandas as pd

def dataframe_to_markdown_chunks(df: pd.DataFrame, chunk_size: int = 50) -> list:
    """
    Splits a DataFrame into row-wise chunks and converts each to Markdown.

    Args:
        df (pd.DataFrame): Input DataFrame (e.g., from Excel).
        chunk_size (int): Number of rows per chunk.

    Returns:
        List[str]: List of Markdown strings.
    """
    chunks = []
    num_rows = len(df)

    for start in range(0, num_rows, chunk_size):
        end = start + chunk_size
        chunk_df = df.iloc[start:end]
        markdown_chunk = chunk_df.to_markdown(index=False)  # omit index if not needed
        chunks.append(markdown_chunk)

    return chunks

import pandas as pd
import numpy as np
from pathlib import Path
import re

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# ---------- CONFIG ----------
EXCEL_PATH = "rent_roll.xlsx"
SHEET_NAME = 0
MAX_HEADER_ROWS = 5  # Number of rows to consider as part of multi-row header
MODEL_NAME = "gemini-1.5-pro-latest"  # or "gemini-1.5-flash-latest"

# ---------- LOAD EXCEL ----------
df_raw = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, header=None)

# ---------- DETECT HEADER ----------
def detect_header(df, max_rows=MAX_HEADER_ROWS):
    for i in range(max_rows):
        row = df.iloc[i].astype(str).str.lower()
        if any(row.str.contains(r'unit|lease|rent|gla|tenant')):
            return i
    return max_rows - 1

header_row = detect_header(df_raw)
header_block = df_raw.iloc[:header_row + 1].fillna("").astype(str)

# ---------- FLATTEN HEADER ----------
def flatten_header(block):
    flat = []
    for col in block.columns:
        parts = [str(block.iloc[i, col]).strip() for i in range(len(block))]
        label = " ".join(filter(None, parts))
        label = re.sub(r'\s+', '_', label).lower()
        flat.append(label)
    return flat

df_data = df_raw.iloc[header_row + 1:].copy()
df_data.columns = flatten_header(header_block)

# Drop empty rows
df_data = df_data.dropna(how="all").reset_index(drop=True)

# Optional: convert date columns to string
for col in df_data.columns:
    if pd.api.types.is_datetime64_any_dtype(df_data[col]):
        df_data[col] = df_data[col].dt.date.astype(str)
    elif df_data[col].dtype == float:
        df_data[col] = df_data[col].round(2)

# ---------- CONVERT TO PIPE-DELIMITED CSV ----------
csv_string = df_data.to_csv(sep="|", index=False)

# ---------- LLM INVOCATION ----------
llm = ChatGoogleGenerativeAI(model=MODEL_NAME)

system_instruction = f"""
Below is a rent roll table from an Excel file. It uses pipe delimiters. The first row contains the column headers.
Ignore footer rows like summaries or totals if present.

Extract the relevant data for each tenant as JSON objects with these fields:
- property
- unit
- market_rent
- current_rent
- lease_start
- lease_end
- gla
- move_in
- move_out

Only include rows with valid unit values.

Return a list of JSON objects like:
[
  {{
    "property": "...",
    "unit": "...",
    "market_rent": 2350.00,
    ...
  }},
  ...
]
"""

prompt = f"""{system_instruction}

```csv
{csv_string}
"""

response = llm.invoke([HumanMessage(content=prompt)])
print(response.content)

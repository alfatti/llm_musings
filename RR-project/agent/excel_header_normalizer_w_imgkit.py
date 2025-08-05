import os
import io
import base64
import pandas as pd
import imgkit
from xlsx2html import xlsx2html
from google.generativeai import GenerativeModel
from tempfile import NamedTemporaryFile

# Initialize Gemini model
model = GenerativeModel("gemini-pro")  # Replace with gemini-pro-vision if needed

# === Step 1: Read Excel ===
def read_excel_raw(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(file_bytes), header=None)

# === Step 2: Render Excel to PNG using HTML conversion ===
def render_excel_preview_high_fidelity(file_bytes: bytes, file_name: str, nrows: int = 20) -> str:
    temp_excel_path = f"/tmp/{file_name}"
    html_path = temp_excel_path.replace(".xlsx", ".html")
    img_path = temp_excel_path.replace(".xlsx", ".png")

    # Write the uploaded Excel file to disk
    with open(temp_excel_path, "wb") as f:
        f.write(file_bytes)

    # Convert to HTML using xlsx2html
    with open(html_path, "w", encoding="utf-8") as f_html:
        xlsx2html(temp_excel_path, f_html, sheet="Sheet1", startrow=0, endrow=nrows)

    # Convert HTML to PNG using imgkit
    imgkit.from_file(html_path, img_path)

    return img_path  # Return the full path to the image

# === Step 3: Convert image to base64 for embedding ===
def image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        image_bytes = f.read()
    return base64.b64encode(image_bytes).decode("utf-8")

# === Step 4: Ask Gemini to write cleaning code ===
def ask_llm_for_cleaning_code(image_b64: str) -> str:
    prompt = f"""
You are given a screenshot of the top of an Excel sheet.
Write Python code using Pandas to:
- Identify and keep only the data body (skip any title/subtitle rows)
- Merge split column names into one row if needed
- Set proper column headers
- Assign the cleaned DataFrame to `df_cleaned`

Assume the raw DataFrame is loaded into `df_raw`.

Here's the image:
![excel_preview](data:image/png;base64,{image_b64})
"""
    response = model.generate_content(prompt)
    return response.text

# === Step 5: Execute LLM-generated cleaning code ===
def apply_cleaning_code(df_raw: pd.DataFrame, code_str: str) -> pd.DataFrame:
    local_scope = {"pd": pd, "df_raw": df_raw.copy()}
    try:
        exec(code_str, {}, local_scope)
        return local_scope.get("df_cleaned", df_raw)
    except Exception as e:
        print(f"Execution failed: {e}")
        return df_raw

# === Step 6: Convert cleaned DF to markdown ===
def df_to_markdown(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)

# === Main Pipeline ===
def process_uploaded_excel(file_bytes: bytes, filename: str) -> str:
    df_raw = read_excel_raw(file_bytes)
    image_path = render_excel_preview_high_fidelity(file_bytes, filename)
    image_b64 = image_to_base64(image_path)
    llm_code = ask_llm_for_cleaning_code(image_b64)
    print("\n=== LLM Generated Code ===\n", llm_code)  # Optional: inspect code
    df_cleaned = apply_cleaning_code(df_raw, llm_code)
    markdown_output = df_to_markdown(df_cleaned)
    return markdown_output

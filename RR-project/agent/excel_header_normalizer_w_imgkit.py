import os
import base64
import pandas as pd
import imgkit
from xlsx2html import xlsx2html
from google.generativeai import GenerativeModel

# Initialize Gemini model (replace with your wrapper if needed)
model = GenerativeModel("gemini-pro")

# === Step 1: Read Excel from path ===
def read_excel_raw(file_path: str) -> pd.DataFrame:
    return pd.read_excel(file_path, header=None)

# === Step 2: Generate high-fidelity screenshot ===
def render_excel_preview_high_fidelity(file_path: str, nrows: int = 20) -> str:
    base = os.path.splitext(os.path.basename(file_path))[0]
    html_path = f"/tmp/{base}_preview.html"
    img_path = f"/tmp/{base}_preview.png"

    # Convert to HTML using xlsx2html
    with open(html_path, "w", encoding="utf-8") as f_html:
        xlsx2html(file_path, f_html, sheet="Sheet1", startrow=0, endrow=nrows)

    # Convert HTML to PNG using imgkit
    imgkit.from_file(html_path, img_path)

    return img_path

# === Step 3: Encode image to base64 ===
def image_to_base64(img_path: str) -> str:
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# === Step 4: Ask Gemini to write header-cleaning code ===
def ask_llm_for_cleaning_code(image_b64: str) -> str:
    prompt = f"""
You are given a screenshot of the top of an Excel sheet.
Write Python code using Pandas to:
- Identify and keep only the data body (skip any title/subtitle rows)
- Merge split column names into one row if needed
- Set proper column headers
- Assign the cleaned DataFrame to `df_cleaned`

Assume the raw DataFrame is already loaded into `df_raw`.

Here is the screenshot:
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
        print(f"[ERROR] LLM-generated code failed: {e}")
        return df_raw

# === Step 6: Convert cleaned DF to markdown ===
def df_to_markdown(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)

# === MAIN PIPELINE ===
def process_excel_from_path(file_path: str) -> str:
    df_raw = read_excel_raw(file_path)
    img_path = render_excel_preview_high_fidelity(file_path)
    img_b64 = image_to_base64(img_path)
    llm_code = ask_llm_for_cleaning_code(img_b64)

    print("\n=== LLM-Generated Pandas Code ===\n")
    print(llm_code)

    df_cleaned = apply_cleaning_code(df_raw, llm_code)
    return df_to_markdown(df_cleaned)

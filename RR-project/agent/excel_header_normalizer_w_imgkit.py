import os
import pandas as pd
import imgkit
import base64
from google.generativeai import GenerativeModel

# === Configure Gemini ===
model = GenerativeModel("gemini-pro")

# === Step 1: Load messy Excel sheet ===
def load_raw_excel(file_path: str, n_preview_rows: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_raw = pd.read_excel(file_path, header=None)
    df_head = df_raw.head(n_preview_rows)
    return df_raw, df_head

# === Step 2: Convert DataFrame head to styled HTML, render as PNG ===
def render_df_head_to_image(df_head: pd.DataFrame, file_path: str) -> str:
    base = os.path.splitext(os.path.basename(file_path))[0]
    html_path = f"/tmp/{base}_preview.html"
    img_path = f"/tmp/{base}_preview.png"

    # Save styled HTML
    html = df_head.style.set_table_attributes('border="1" class="dataframe table"').to_html()
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    # Convert to PNG
    imgkit.from_file(html_path, img_path)
    return img_path

# === Step 3: Encode screenshot to base64 ===
def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# === Step 4: Prompt Gemini to generate cleaning code ===
def prompt_llm_with_image(image_b64: str) -> str:
    prompt = f"""
You are given a screenshot of the top of an Excel sheet.
Write Python code using Pandas to:
- Remove title or subtitle rows
- Merge split header rows into one row
- Set correct column headers
- Assign the cleaned DataFrame to `df_cleaned`

Assume the messy DataFrame is already loaded as `df_raw`.

Screenshot:
![excel_preview](data:image/png;base64,{image_b64})
"""
    response = model.generate_content(prompt)
    return response.text

# === Step 5: Run Gemini's cleaning code ===
def execute_cleaning_code(df_raw: pd.DataFrame, code_str: str) -> pd.DataFrame:
    local_vars = {"pd": pd, "df_raw": df_raw.copy()}
    try:
        exec(code_str, {}, local_vars)
        return local_vars.get("df_cleaned", df_raw)
    except Exception as e:
        print(f"[ERROR] Failed to execute LLM code: {e}")
        return df_raw

# === Step 6: Output as markdown ===
def to_markdown(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)

# === MAIN WRAPPER ===
def clean_excel_headers(file_path: str) -> str:
    print(f"[INFO] Loading: {file_path}")
    df_raw, df_head = load_raw_excel(file_path)

    print("[INFO] Rendering image...")
    img_path = render_df_head_to_image(df_head, file_path)

    print("[INFO] Encoding image to base64...")
    img_b64 = encode_image_to_base64(img_path)

    print("[INFO] Prompting Gemini to generate cleaning code...")
    llm_code = prompt_llm_with_image(img_b64)

    print("\n=== LLM-Generated Cleaning Code ===\n")
    print(llm_code)

    print("[INFO] Executing cleaning code...")
    df_cleaned = execute_cleaning_code(df_raw, llm_code)

    print("[INFO] Converting cleaned table to markdown...")
    return to_markdown(df_cleaned)

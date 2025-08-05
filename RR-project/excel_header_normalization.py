import pandas as pd
from openpyxl import load_workbook
from PIL import ImageGrab
import io
import base64
import requests
import json
import os

# === CONFIG ===
RR_PATH = "your_rent_roll.xlsx"
GEMINI_ENDPOINT = "https://your_gemini_endpoint_here"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")  # or hardcode for testing

# === STEP 1: Load Excel Preview and Screenshot ===
def get_excel_top_image(excel_path, n_rows=20, save_to_disk=True):
    # Load preview
    df_preview = pd.read_excel(excel_path, sheet_name=0, header=None, nrows=n_rows)

    # Render as matplotlib table
    fig, ax = plt.subplots(figsize=(12, 0.5 * n_rows))
    ax.axis('off')
    table = ax.table(cellText=df_preview.values,
                     colLabels=None,
                     loc='center',
                     cellLoc='left')

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.2)

    # Save to BytesIO
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    img_bytes = buf.getvalue()

    # Save to disk with Excel name prepended
    if save_to_disk:
        basename = os.path.splitext(os.path.basename(excel_path))[0]
        output_png = f"{basename}_preview.png"
        with open(output_png, "wb") as f:
            f.write(img_bytes)
        print(f"Image saved to: {output_png}")

    return img_bytes

# === STEP 2: Encode Image for Gemini ===
def encode_image_to_base64(img_bytes):
    return base64.b64encode(img_bytes).decode('utf-8')

# === STEP 3: Construct Prompt ===
def build_prompt(encoded_img):
    return {
        "contents": [{
            "parts": [
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": encoded_img
                    }
                },
                {
                    "text": (
                        "The image above shows the top of a rent roll Excel file. "
                        "Write Python pandas code to 'normalize' the schema. "
                        "Normalization means:\n"
                        "1. Skip the first few rows containing titles, subtitles, etc. Keep only the actual data table body.\n"
                        "2. If column names are split across two rows, conjoin them (e.g., row1='Scheduled', row2='Rent' → 'Scheduled Rent').\n"
                        "Assume the file is already read using `pd.read_excel(RR_PATH, header=None)`.\n"
                        "Return just the pandas code that cleans and normalizes this DataFrame. "
                        "Do not re-read the Excel file inside your code."
                    )
                }
            ]
        }]
    }

# === STEP 4: Send to Gemini ===
def query_gemini(prompt_json):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GEMINI_API_KEY}"
    }
    response = requests.post(GEMINI_ENDPOINT, headers=headers, json=prompt_json)
    response.raise_for_status()
    return response.json()["candidates"][0]["content"]["parts"][0]["text"]



def strip_markdown_code_fence(code_str):
    lines = code_str.strip().splitlines()
    if lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)

# === STEP 5: Execute the Returned Code ===
def execute_normalization_script(script, df_raw):
    local_vars = {'pd': pd, 'df': df_raw.copy()}
    exec(script, {}, local_vars)
    return local_vars.get("df", df_raw)


import matplotlib.pyplot as plt
from io import BytesIO
import os

def save_dataframe_preview_image(df, excel_path, n_rows=20):
    """
    Saves a PNG image of the top n_rows of a DataFrame.
    The image filename is derived from the Excel file name.

    Args:
        df (pd.DataFrame): The normalized DataFrame.
        excel_path (str): Path to the original Excel file (used for naming).
        n_rows (int): Number of top rows to include in the image.
    """
    df_preview = df.head(n_rows)

    fig, ax = plt.subplots(figsize=(12, 0.5 * n_rows))
    ax.axis('off')
    table = ax.table(cellText=df_preview.values,
                     colLabels=df_preview.columns,
                     loc='center',
                     cellLoc='left')

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.2)

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)

    # Create output filename
    basename = os.path.splitext(os.path.basename(excel_path))[0]
    output_png = f"{basename}_normalized_preview.png"
    with open(output_png, "wb") as f:
        f.write(buf.getvalue())

    print(f"Normalized DataFrame preview saved to: {output_png}")


# === MAIN ===
if __name__ == "__main__":
    # Load the raw data
    df_raw = pd.read_excel(RR_PATH, sheet_name=0, header=None)

    # Prepare image and prompt
    img_bytes = get_excel_top_image(RR_PATH)
    img_b64 = encode_image_to_base64(img_bytes)
    prompt = build_prompt(img_b64)

    # Get code from Gemini
    code = query_gemini(prompt)
    print("Gemini returned code:\n", code)

    # Execute the normalization
    df_normalized = execute_normalization_script(code, df_raw)

    # Show result
    print(df_normalized.head())

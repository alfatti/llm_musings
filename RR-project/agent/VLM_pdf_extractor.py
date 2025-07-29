import io
import pandas as pd
from pdf2image import convert_from_path
from PIL import Image
import google.generativeai as genai
from markdown import markdown
from bs4 import BeautifulSoup

# -------------------------------
# 1. Gemini Setup
# -------------------------------
genai.configure(api_key="YOUR_API_KEY")  # Replace with your Gemini API Key
model = genai.GenerativeModel("gemini-pro-vision")

# -------------------------------
# 2. Convert PDF to in-memory images
# -------------------------------
pdf_path = "rent_roll_sample_final.pdf"
images = convert_from_path(pdf_path, dpi=300)

# -------------------------------
# 3. Prompt Template
# -------------------------------
PROMPT = """
You are a rent roll extraction assistant.
The image contains a rent roll table used in commercial real estate.
Extract all tables you can see and return them in Markdown format.
- Include all rows and units even if data is missing
- Preserve column headers as seen
- Return only Markdown tables (no explanations)
"""

# -------------------------------
# 4. Helper: Convert Markdown → DataFrame
# -------------------------------
def markdown_to_df(markdown_str):
    html = markdown(markdown_str)
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    dfs = []
    for table in tables:
        headers = [th.text.strip() for th in table.find_all("th")]
        rows = []
        for row in table.find_all("tr")[1:]:
            cells = [td.text.strip() for td in row.find_all("td")]
            rows.append(cells)
        df = pd.DataFrame(rows, columns=headers if headers else None)
        dfs.append(df)
    return dfs

# -------------------------------
# 5. Run Gemini on Each Image
# -------------------------------
all_dfs = []

for idx, img in enumerate(images):
    print(f"Processing page {idx+1}")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    response = model.generate_content([PROMPT, img_byte_arr], stream=False)
    md = response.text

    try:
        dfs = markdown_to_df(md)
        all_dfs.extend(dfs)
    except Exception as e:
        print(f"Markdown parsing failed on page {idx+1}: {e}")
        print(md)

# -------------------------------
# 6. Combine or Inspect
# -------------------------------
for i, df in enumerate(all_dfs):
    print(f"\n--- Table {i+1} ---")
    print(df.head())

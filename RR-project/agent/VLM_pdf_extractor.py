import base64
from pdf2image import convert_from_path
from PIL import Image
import google.generativeai as genai
from markdown import markdown
from bs4 import BeautifulSoup
import pandas as pd
from io import BytesIO

PROMPT_ADDENDUM =
"""You are a rent roll extraction engine.

Input: A page from a rent roll document (as an image).
Output: Markdown tables ONLY.

Guidelines:
- Return one or more Markdown tables that represent the structured information on the page.
- Do NOT include any descriptive text.
- Do NOT explain what the table is.
- Do NOT preface your answer with any summary.
- Do NOT say "Here is the table below".
- Just return Markdown tables directly.
"""

# -------------------------------
# 1. Gemini Setup
# -------------------------------
genai.configure(api_key="YOUR_API_KEY")  # Replace with your Gemini API Key
model = genai.GenerativeModel("gemini-pro-vision")

# -------------------------------
# 2. Prompt Template
# -------------------------------
PROMPT = """
The image is a page from a rent roll document used for commercial real estate underwriting.
Extract all visible tables and output them strictly in Markdown format.

Guidelines:
- Include all rows (even if some columns are empty)
- Preserve header structure
- If multiple tables, output them sequentially, separated by newlines
- No explanations, just the tables
"""

# -------------------------------
# 3. Helper: Encode image to base64
# -------------------------------
def image_to_base64(pil_img: Image.Image) -> str:
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return encoded

# -------------------------------
# 4. Helper: Convert Markdown → List of DataFrames
# -------------------------------
def markdown_to_df(md: str):
    html = markdown(md)
    soup = BeautifulSoup(html, "html.parser")
    dfs = []
    for table in soup.find_all("table"):
        headers = [th.text.strip() for th in table.find_all("th")]
        rows = []
        for row in table.find_all("tr")[1:]:
            cells = [td.text.strip() for td in row.find_all("td")]
            rows.append(cells)
        df = pd.DataFrame(rows, columns=headers if headers else None)
        dfs.append(df)
    return dfs

# -------------------------------
# 5. Load PDF and Run Extraction
# -------------------------------
pdf_path = "rent_roll_sample_final.pdf"
images = convert_from_path(pdf_path, dpi=300)

all_dfs = []

for idx, img in enumerate(images):
    print(f"🔍 Processing page {idx + 1}")
    
    b64_image = image_to_base64(img)

    try:
        response = model.generate_content(
            [PROMPT, {"mime_type": "image/png", "data": b64_image}],
            stream=False
        )
        markdown_output = response.text
        dfs = markdown_to_df(markdown_output)
        all_dfs.extend(dfs)
    except Exception as e:
        print(f"❌ Error on page {idx + 1}: {e}")

# -------------------------------
# 6. View or Combine Output
# -------------------------------
for i, df in enumerate(all_dfs):
    print(f"\n📄 Table {i+1}")
    print(df.head())

#1) Imports & helpers (PDF→image, Excel→CSV/Markdown, pair builder)
# Cell 1 — Setup & helpers
import base64, io, os, textwrap, json, requests
from typing import List, Tuple, Optional

# PDF → PNG (base64). Tries PyMuPDF (fitz) first; falls back to pdf2image if available.
def pdf_page_to_base64_png(pdf_path: str, page_index: int = 0, dpi: int = 200, max_width: int = 1600) -> str:
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        page = doc[page_index]
        # scale by DPI
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = pix.tobytes("png")
    except Exception:
        # fallback to pdf2image
        from pdf2image import convert_from_path
        images = convert_from_path(pdf_path, dpi=dpi, first_page=page_index + 1, last_page=page_index + 1)
        buf = io.BytesIO()
        images[0].save(buf, format="PNG")
        img = buf.getvalue()

    # optional downscale to cap width
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(img))
        if im.width > max_width:
            ratio = max_width / im.width
            im = im.resize((max_width, int(im.height * ratio)))
            out = io.BytesIO()
            im.save(out, format="PNG")
            img = out.getvalue()
    except Exception:
        pass

    return base64.b64encode(img).decode("utf-8")


# Excel/CSV → CSV string with fixed header order
def dataframe_to_csv_string(df, header_order: Optional[List[str]] = None) -> str:
    import pandas as pd

    if header_order:
        # keep only known columns (in order); include missing as empty
        for col in header_order:
            if col not in df.columns:
                df[col] = pd.NA
        df = df[header_order]
    # normalize dtypes a bit
    df = df.where(df.notna(), "")
    return df.to_csv(index=False)

def excel_extract_to_csv(
    xlsx_path: str,
    sheet: Optional[str] = None,
    header_order: Optional[List[str]] = None
) -> str:
    import pandas as pd
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    return dataframe_to_csv_string(df, header_order)


# Build a “few-shot turn” from (image_b64, gold_csv)
def make_exemplar_turn(image_b64_png: str, gold_csv: str):
    user = {
        "role": "user",
        "parts": [
            {"text": "Example rent roll page image:"},
            {"inline_data": {"mime_type": "image/png", "data": image_b64_png}},
            {"text": "Expected CSV for that page (emit EXACTLY this structure in your answer):\n" + gold_csv}
        ],
    }
    model = {
        "role": "model",
        "parts": [{"text": gold_csv}]
    }
    return user, model
#2) Write the instruction prompt (separate cell)

# Cell 2 — Prompt/instructions
OUTPUT_COLUMNS = [
    "Unit","Unit Type","SqFt","Resident","Status",
    "Market Rent","Concession Amount","Is Concession",
    "Move In","Move Out"
]

BASE_INSTRUCTIONS = f"""
You are a precise rent-roll table extractor.

TASK
- Given a single-page rent roll image, extract a row-wise table with columns:
  {', '.join(OUTPUT_COLUMNS)}.
- The source forms vary widely. Use visual + text cues, merged cells, side labels, multi-row headers, and legends.
- Normalize:
  - Dates → ISO (YYYY-MM-DD) when present; else blank.
  - Currency → numbers without $ or commas (e.g., 1234.56). Leave blank if not shown.
  - Booleans like “Is Concession” → true/false (lowercase).
  - Missing values → empty cell.

STRICT OUTPUT
- Output ONLY CSV with header row EXACTLY:
  {','.join(OUTPUT_COLUMNS)}
- No explanations, no extra columns, no code-fences.
- One row per unit present on the page (deduplicate if repeated).

QUALITY RULES
- Prefer explicit values on the page over inferred ones.
- If a field is not visible, leave the cell empty (do NOT hallucinate).
- Validate row counts against visible unit identifiers.
""".strip()
#3) Build the contents with ~6 exemplars
# Cell 3 — Load exemplars (your files) and assemble contents
# Fill these with your training examples: each PDF page + its gold extract (Excel saved separately).
# Examples assume first page in each PDF and single-sheet Excel "gold" per exemplar.
example_pdf_paths = [
    "examples/rr_01.pdf",
    "examples/rr_02.pdf",
    "examples/rr_03.pdf",
    "examples/rr_04.pdf",
    "examples/rr_05.pdf",
    "examples/rr_06.pdf",
]
example_gold_excel_paths = [
    "examples/rr_01_gold.xlsx",
    "examples/rr_02_gold.xlsx",
    "examples/rr_03_gold.xlsx",
    "examples/rr_04_gold.xlsx",
    "examples/rr_05_gold.xlsx",
    "examples/rr_06_gold.xlsx",
]

# Convert exemplars to (image_b64, csv) and then to user/model turns
contents = []
for pdf_path, gold_xlsx in zip(example_pdf_paths, example_gold_excel_paths):
    img_b64 = pdf_page_to_base64_png(pdf_path, page_index=0, dpi=200)
    gold_csv = excel_extract_to_csv(gold_xlsx, header_order=OUTPUT_COLUMNS)
    u, m = make_exemplar_turn(img_b64, gold_csv)
    contents.extend([u, m])

# Prepare current page to extract (fill these when you run for real)
CURRENT_PDF_PATH = "inputs/new_rentroll_page.pdf"
current_img_b64 = pdf_page_to_base64_png(CURRENT_PDF_PATH, page_index=0, dpi=200)

# Add the current user query turn
contents.append({
    "role": "user",
    "parts": [
        {"text": BASE_INSTRUCTIONS + "\n\nNow extract the CSV for the following page."},
        {"inline_data": {"mime_type": "image/png", "data": current_img_b64}},
    ],
})


#4) Make the REST call (Gemini Generative Language API)
# Cell 4 — REST call (parameterize MODEL_ID/ENDPOINT/API_KEY)
API_KEY = os.getenv("GEMINI_API_KEY")  # or paste securely
MODEL_ID = os.getenv("GEMINI_MODEL_ID", "gemini-2.5-pro")  # use your exact model name
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_ID}:generateContent"

request_body = {
    "contents": contents,
    "generationConfig": {
        "temperature": 0.0,
        "topP": 0.1,
        "maxOutputTokens": 4096,
        # Optional: “responseMimeType”: “text/csv” (supported in newer endpoints)
    },
}

headers = {
    "Content-Type": "application/json",
    "x-goog-api-key": API_KEY,
}

resp = requests.post(GEMINI_ENDPOINT, headers=headers, data=json.dumps(request_body), timeout=120)
resp.raise_for_status()
data = resp.json()

# Extract text
try:
    reply_text = data["candidates"][0]["content"]["parts"][0]["text"]
except Exception:
    # Fallback parse if the structure differs
    reply_text = json.dumps(data, indent=2)

print(reply_text)

#5) (Optional) Parse the CSV reply into a DataFrame and save
# Cell 5 — Convert CSV reply → DataFrame → Excel tab
import pandas as pd

csv_text = reply_text.strip()
df_out = pd.read_csv(io.StringIO(csv_text))
display(df_out.head())

# Save to a workbook or a new tab
OUT_XLSX = "outputs/rentroll_extract.xlsx"
with pd.ExcelWriter(OUT_XLSX, engine="openpyxl", mode="w") as xw:
    df_out.to_excel(xw, sheet_name="RENT_ROLL_EXTRACT", index=False)

OUT_XLSX

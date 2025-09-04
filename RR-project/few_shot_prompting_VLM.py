# === Cell 1: Config & imports ===
import os, io, json, time, base64, textwrap, pathlib
from typing import List, Tuple, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd

# --- Gemini REST config (set your vars here or via environment) ---
API_KEY   = os.getenv("GEMINI_API_KEY", "PUT_YOUR_KEY_HERE")
MODEL_ID  = os.getenv("GEMINI_MODEL_ID", "gemini-2.5-pro")
ENDPOINT  = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_ID}:generateContent"

# Output folder
OUTPUT_DIR = pathlib.Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === Cell 2: Prompt & schema ===
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

NORMALIZATION
- Dates → ISO (YYYY-MM-DD); else blank.
- Currency → numbers without $ or commas (e.g., 1234.56); else blank.
- Booleans like “Is Concession” → true/false (lowercase).
- Missing values → empty cell.

STRICT OUTPUT
- Output ONLY CSV with header row EXACTLY:
  {','.join(OUTPUT_COLUMNS)}
- No explanations, no extra columns, no code fences.
- One row per unit present on the page (deduplicate if repeated).

QUALITY RULES
- Prefer explicit values on the page over inferred ones.
- If a field is not visible, leave the cell empty (do NOT hallucinate).
- Validate row counts against visible unit identifiers.
""".strip()

# === Cell 3: Helpers ===
import io

def _resize_png_if_needed(png_bytes: bytes, max_width: int = 1600) -> bytes:
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(png_bytes))
        if im.width > max_width:
            ratio = max_width / im.width
            im = im.resize((max_width, int(im.height * ratio)))
            out = io.BytesIO()
            im.save(out, format="PNG")
            return out.getvalue()
        return png_bytes
    except Exception:
        return png_bytes

def pdf_to_base64_pages(pdf_path: str, dpi: int = 200, max_width: int = 1600) -> List[str]:
    """
    Render ALL pages of a PDF to PNG (base64). Tries PyMuPDF; falls back to pdf2image.
    Returns list of base64 strings, one per page.
    """
    pages_b64 = []
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for i in range(len(doc)):
            page = doc[i]
            pix = page.get_pixmap(matrix=mat, alpha=False)
            png = pix.tobytes("png")
            png = _resize_png_if_needed(png, max_width=max_width)
            pages_b64.append(base64.b64encode(png).decode("utf-8"))
    except Exception:
        # fallback to pdf2image
        from pdf2image import convert_from_path
        images = convert_from_path(pdf_path, dpi=dpi)
        for im in images:
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            png = _resize_png_if_needed(buf.getvalue(), max_width=max_width)
            pages_b64.append(base64.b64encode(png).decode("utf-8"))
    return pages_b64

def read_table_file(file_path: str, sheet: Optional[object] = None) -> pd.DataFrame:
    """
    Robust loader for gold extracts:
      - .xlsx/.xls: returns a single DataFrame (defaults to first sheet if sheet is None)
      - .csv: returns DataFrame via read_csv
      - .json: expects array-of-objects and returns DataFrame
    """
    ext = pathlib.Path(file_path).suffix.lower()
    if ext in (".xlsx", ".xls"):
        # Default to first sheet instead of None (which returns dict-of-DFs)
        sheet_name = 0 if sheet is None else sheet
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        # In case the caller passes a list of sheets, Pandas returns a dict
        if isinstance(df, dict):
            # Pick the first sheet in the dict
            df = next(iter(df.values()))
        return df
    elif ext == ".csv":
        return pd.read_csv(file_path)
    elif ext == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported table file type: {ext}")

def dataframe_to_csv_string(df: pd.DataFrame, header_order: Optional[List[str]] = None) -> str:
    if isinstance(df, dict):
        raise TypeError(
            "Expected a pandas DataFrame but got a dict. "
            "This usually means read_excel returned multiple sheets. "
            "Pass sheet=... or ensure a single sheet is selected."
        )
    df = df.copy()
    if header_order:
        for col in header_order:
            if col not in df.columns:
                df[col] = pd.NA
        df = df[header_order]
    df = df.where(df.notna(), "")
    return df.to_csv(index=False)

def excel_extract_to_csv(
    table_path: str,
    sheet: Optional[object] = None,
    header_order: Optional[List[str]] = None
) -> str:
    """
    Load an Excel/CSV/JSON 'gold' extract and normalize to CSV with the requested headers.
    """
    df = read_table_file(table_path, sheet=sheet)
    return dataframe_to_csv_string(df, header_order)


# === Cell 4: Load exemplars ===
# Fill these paths with your few-shot training examples:
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

# Convert exemplars → base64 page image (first page) + gold CSV aligned to schema
exemplars: List[Tuple[str, str]] = []
for pdf_path, gold_xlsx in zip(example_pdf_paths, example_gold_excel_paths):
    pages_b64 = pdf_to_base64_pages(pdf_path, dpi=200)
    if not pages_b64:
        raise RuntimeError(f"No pages found in exemplar PDF: {pdf_path}")
    gold_csv = excel_extract_to_csv(gold_xlsx, header_order=OUTPUT_COLUMNS)
    exemplars.append((pages_b64[0], gold_csv))

len(exemplars), "exemplars loaded"

# === Cell 5: Gemini call helpers ===

def gemini_generate_csv(contents: List[dict], timeout_s: int = 120, max_retries: int = 3, backoff: float = 1.5) -> str:
    """
    Calls Gemini once with provided contents; returns text (CSV expected).
    Retries simple transient failures with exponential backoff.
    """
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": API_KEY,
    }
    body = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0.0,
            "topP": 0.1,
            "maxOutputTokens": 4096,
            # "responseMimeType": "text/csv",  # Uncomment if your endpoint supports it reliably
        },
    }

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(ENDPOINT, headers=headers, data=json.dumps(body), timeout=timeout_s)
            resp.raise_for_status()
            data = resp.json()
            # Typical path:
            reply = data["candidates"][0]["content"]["parts"][0]["text"]
            return strip_code_fences(reply)
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(backoff ** attempt)
            else:
                raise last_err

def extract_page_csv_from_image_b64(
    image_b64: str,
    exemplars: List[Tuple[str, str]],
    instructions: str
) -> str:
    contents = build_contents_for_page(exemplars, instructions, image_b64)
    return gemini_generate_csv(contents)
# === Cell 6: Batch extraction for a multipage PDF ===
def process_pdf_multipage_to_csvs(
    pdf_path: str,
    exemplars: List[Tuple[str, str]],
    instructions: str,
    max_workers: int = 4
) -> Dict[int, str]:
    """
    Shreds a multi-page PDF into images, then parallel-calls Gemini for each page.
    Returns {page_index: csv_text}
    """
    pages_b64 = pdf_to_base64_pages(pdf_path, dpi=200)
    if not pages_b64:
        raise RuntimeError(f"No pages found in: {pdf_path}")

    results: Dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(extract_page_csv_from_image_b64, b64, exemplars, instructions): idx
            for idx, b64 in enumerate(pages_b64)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                csv_text = fut.result()
                results[idx] = csv_text
            except Exception as e:
                results[idx] = f"__ERROR__: {e}"

    # ensure results in page order
    return dict(sorted(results.items(), key=lambda kv: kv[0]))


def combine_page_csvs_to_dataframe(
    page_csvs: Dict[int, str],
    add_page_col: bool = True
) -> pd.DataFrame:
    """
    Combine per-page CSV strings into a single DataFrame.
    Adds a 'PageIndex' column (0-based) when add_page_col=True.
    Skips pages that errored or returned empty.
    """
    frames = []
    for idx, csv_text in page_csvs.items():
        if not csv_text or csv_text.startswith("__ERROR__"):
            continue
        try:
            df = pd.read_csv(io.StringIO(csv_text))
            if add_page_col:
                df.insert(0, "PageIndex", idx)
            # Enforce output columns if present
            for col in OUTPUT_COLUMNS:
                if col not in df.columns:
                    df[col] = pd.NA
            df = df[(["PageIndex"] if add_page_col else []) + OUTPUT_COLUMNS]
            frames.append(df)
        except Exception:
            # keep going even if a single page produces malformed CSV
            continue
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=(["PageIndex"] if add_page_col else []) + OUTPUT_COLUMNS)
# === Cell 7: Execute on your multipage PDF ===
MULTIPAGE_PDF = "inputs/new_rentroll_multipage.pdf"  # <-- point to your file

page_csvs = process_pdf_multipage_to_csvs(
    pdf_path=MULTIPAGE_PDF,
    exemplars=exemplars,
    instructions=BASE_INSTRUCTIONS,
    max_workers=6,   # tune this based on your rate limits & CPU
)

# Save per-page CSV files
base = pathlib.Path(MULTIPAGE_PDF).stem
for idx, csv_text in page_csvs.items():
    out = OUTPUT_DIR / f"{base}_page{idx:03d}.csv"
    out.write_text(csv_text if csv_text else "", encoding="utf-8")

# Combine and save to Excel
df_all = combine_page_csvs_to_dataframe(page_csvs, add_page_col=True)
display(df_all.head())

xlsx_out = OUTPUT_DIR / f"{base}_extract.xlsx"
with pd.ExcelWriter(xlsx_out, engine="openpyxl", mode="w") as xw:
    df_all.to_excel(xw, sheet_name="RENT_ROLL_EXTRACT", index=False)

xlsx_out

# Notes & tuning
# Parallelization: Adjust max_workers to respect your org’s Gemini quotas. If you hit 429s, reduce workers or add stronger backoff in gemini_generate_csv.
# Reliability: If you see occasional code-fenced replies, the strip_code_fences() helper will clean them. You can also set "responseMimeType": "text/csv" (commented) if your endpoint supports it.
# Schema drift: Keep OUTPUT_COLUMNS in sync across (a) prompt, (b) exemplar CSVs, and (c) combiner logic.
# Exemplars: 4–8 high-quality, diverse examples typically work best. You can rotate them by property type/layout to improve generalization.
# If you want, I can add an error review tab (pages with malformed CSVs + raw text) or a rate-limit-aware queue (token bucket) for friendlier scaling.


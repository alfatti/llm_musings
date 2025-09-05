
# === Cell 1: Config & imports ===
import os, io, json, time, base64, pathlib
from typing import List, Tuple, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd

# ---- Vertex AI (fill these or set via environment) ----
PROJECT_ID = os.getenv("VERTEX_PROJECT_ID", "YOUR_PROJECT_ID")
LOCATION   = os.getenv("VERTEX_LOCATION",   "us-central1")
MODEL_ID   = os.getenv("VERTEX_MODEL_ID",   "gemini-2.5-flash")
TOKEN      = os.getenv("VERTEX_TOKEN",      "PASTE_YOUR_BEARER_TOKEN_HERE")

VERTEX_HOST = f"https://{LOCATION}-aiplatform.googleapis.com"
ENDPOINT    = f"{VERTEX_HOST}/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL_ID}:generateContent"

OUTPUT_DIR = pathlib.Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# === Cell 2: Prompt & schema ===
OUTPUT_COLUMNS = [
    "Unit","Unit Type","SqFt","Resident","Status",
    "Market Rent","Concession Amount","Is Concession",
    "Move In","Move Out"
]

REQUIRED_HEADER = ",".join(OUTPUT_COLUMNS)

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
  {REQUIRED_HEADER}
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
        from pdf2image import convert_from_path
        images = convert_from_path(pdf_path, dpi=dpi)
        for im in images:
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            png = _resize_png_if_needed(buf.getvalue(), max_width=max_width)
            pages_b64.append(base64.b64encode(png).decode("utf-8"))
    return pages_b64

import pandas as pd

import os, json
import pandas as pd
from typing import Union, List, Optional

def to_csv_string_flexible(
    obj: Union[pd.DataFrame, dict, List[dict], str],
    header_order: Optional[List[str]] = None,
    sheet: Optional[str] = None,
) -> str:
    """
    Convert various inputs to CSV text with optional enforced header order.
    Supported obj types:
      - pandas.DataFrame
      - dict (one row)
      - list[dict] (multiple rows)
      - str path: .xlsx/.xls, .csv, .json
        * Excel: reads `sheet` if provided, else first sheet
        * CSV: pandas.read_csv
        * JSON: either a dict or list[dict] in the file
    """
    # Normalize to DataFrame
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()

    elif isinstance(obj, dict):
        df = pd.DataFrame([obj])

    elif isinstance(obj, list) and all(isinstance(x, dict) for x in obj):
        df = pd.DataFrame(obj)

    elif isinstance(obj, str):
        path = obj
        ext = os.path.splitext(path)[1].lower()
        if ext in (".xlsx", ".xls"):
            df = pd.read_excel(path, sheet_name=sheet)
        elif ext == ".csv":
            df = pd.read_csv(path)
        elif ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            elif isinstance(data, list) and all(isinstance(x, dict) for x in data):
                df = pd.DataFrame(data)
            else:
                raise TypeError(f"JSON must be dict or list[dict], got {type(data)} in {path}")
        else:
            raise TypeError(f"Unsupported file extension '{ext}' for path: {path}")
    else:
        raise TypeError(f"Unsupported exemplar type: {type(obj)}")

    # Enforce header order if provided
    if header_order:
        for col in header_order:
            if col not in df.columns:
                df[col] = pd.NA
        df = df[header_order]

    # Clean NaNs → empty string
    df = df.where(df.notna(), "")
    return df.to_csv(index=False)


    # enforce header order if provided
    if header_order:
        for col in header_order:
            if col not in df.columns:
                df[col] = pd.NA
        df = df[header_order]

    # clean NaNs → empty string
    df = df.where(df.notna(), "")
    return df.to_csv(index=False)


def excel_extract_to_csv(xlsx_path: str, sheet: Optional[str] = None, header_order: Optional[List[str]] = None) -> str:
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    return dataframe_to_csv_string(df, header_order)

# --- Replace your exemplar/content builders with these ---

def _make_exemplar_turn(image_b64_png: str, gold_csv: str) -> tuple[dict, dict]:
    user = {
        "role": "user",
        "parts": [
            {"text": "Example rent roll page image:"},
            {"inlineData": {"mimeType": "image/png", "data": image_b64_png}},  # <-- camelCase
            {"text": "Expected CSV for that page (emit EXACTLY this structure in your answer):\n" + gold_csv}
        ],
    }
    model = {"role": "model", "parts": [{"text": gold_csv}]}
    return user, model

def _build_contents_for_page_inline(
    exemplars: list[tuple[str, str]],
    instructions: str,
    page_image_b64: str
) -> list[dict]:
    contents: list[dict] = []
    for img_b64, csv in exemplars:
        u, m = _make_exemplar_turn(img_b64, csv)
        contents.extend([u, m])
    contents.append({
        "role": "user",
        "parts": [
            {"text": instructions + "\n\nNow extract the CSV for the following page."},
            {"inlineData": {"mimeType": "image/png", "data": page_image_b64}},  # <-- camelCase
        ],
    })
    return contents

def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```") and t.endswith("```"):
        inner = t.strip("`")
        # remove leading language tag if present
        if "\n" in inner:
            first, rest = inner.split("\n", 1)
            return rest.strip()
    return t

def _collect_text_from_vertex_response(data: dict) -> str:
    """
    Robustly gather all text parts from the first candidate.
    Raises on block reasons.
    """
    # Blocked?
    pf = data.get("promptFeedback", {})
    if pf.get("blockReason"):
        raise RuntimeError(f"Prompt blocked: {pf.get('blockReason')} — {pf.get('safetyRatings')}")

    cands = data.get("candidates", [])
    if not cands:
        raise RuntimeError(f"No candidates in response: {json.dumps(data)[:400]}")

    cand = cands[0]
    # Candidate-level block?
    if cand.get("finishReason") == "SAFETY":
        raise RuntimeError(f"Response blocked by safety: {cand.get('safetyRatings')}")
    content = cand.get("content", {})
    parts = content.get("parts", [])
    if not parts:
        raise RuntimeError(f"No content.parts in response: {json.dumps(data)[:400]}")

    texts = []
    for p in parts:
        if "text" in p and isinstance(p["text"], str):
            texts.append(p["text"])
    if not texts:
        raise RuntimeError(f"No text parts in response: {json.dumps(data)[:400]}")

    return _strip_code_fences("\n".join(texts)).strip()

def _looks_like_valid_csv(csv_text: str) -> bool:
    head = csv_text.splitlines()[:1]
    return len(head) == 1 and head[0].strip() == REQUIRED_HEADER

# === Cell 4: Load exemplars ===
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

exemplars: List[Tuple[str, str]] = []
for pdf_path, gold_xlsx in zip(example_pdf_paths, example_gold_excel_paths):
    pages_b64 = pdf_to_base64_pages(pdf_path, dpi=200)
    if not pages_b64:
        raise RuntimeError(f"No pages found in exemplar PDF: {pdf_path}")
    gold_csv = excel_extract_to_csv(gold_xlsx, header_order=OUTPUT_COLUMNS)
    exemplars.append((pages_b64[0], gold_csv))

len(exemplars), "exemplars loaded"

# === Cell 5: Vertex call + page extractor ===

# --- Replace your Vertex call helper with this ---

def vertex_generate_csv(contents: list[dict], timeout_s: int = 120, max_retries: int = 3, backoff: float = 1.7) -> str:
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json; charset=utf-8",
    }
    body = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0.0,
            "topP": 0.1,
            "maxOutputTokens": 4096,
            # If your org supports it, you can keep this; otherwise comment it out:
            # "responseMimeType": "text/csv",
        },
    }

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(ENDPOINT, headers=headers, data=json.dumps(body), timeout=timeout_s)
            # Helpful diagnostics for 4xx/5xx:
            if resp.status_code >= 400:
                # Write the server message so it shows up in your *_errors.txt
                raise RuntimeError(f"{resp.status_code} {resp.reason} — {resp.text[:500]}")
            data = resp.json()
            text = _collect_text_from_vertex_response(data)
            return text
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(backoff ** attempt)
            else:
                raise last_err


def extract_page_csv_from_image_b64(
    image_b64: str,
    exemplars: list[tuple[str, str]],
    instructions: str
) -> str:
    contents = _build_contents_for_page_inline(exemplars, instructions, image_b64)
    csv_text = vertex_generate_csv(contents)
    # Final guard: ensure header is exact; otherwise mark as error to avoid "parts" or JSON blobs becoming CSV
    if not _looks_like_valid_csv(csv_text):
        raise RuntimeError(f"Model did not return a valid CSV with required header. Got head: {csv_text.splitlines()[:2]}")
    return csv_text

# === Cell 6: Batch processing & combine ===
def process_pdf_multipage_to_csvs(
    pdf_path: str,
    exemplars: List[Tuple[str, str]],
    instructions: str,
    max_workers: int = 4
) -> Dict[int, str]:
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
    return dict(sorted(results.items(), key=lambda kv: kv[0]))

def combine_page_csvs_to_dataframe(
    page_csvs: Dict[int, str],
    add_page_col: bool = True
) -> pd.DataFrame:
    frames = []
    for idx, csv_text in page_csvs.items():
        if not csv_text or csv_text.startswith("__ERROR__"):
            continue
        try:
            df = pd.read_csv(io.StringIO(csv_text))
            if add_page_col:
                df.insert(0, "PageIndex", idx)
            for col in OUTPUT_COLUMNS:
                if col not in df.columns:
                    df[col] = pd.NA
            df = df[(["PageIndex"] if add_page_col else []) + OUTPUT_COLUMNS]
            frames.append(df)
        except Exception:
            continue
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=(["PageIndex"] if add_page_col else []) + OUTPUT_COLUMNS)

# === Cell 7: Execute on a multipage PDF ===
MULTIPAGE_PDF = "inputs/new_rentroll_multipage.pdf"  # <-- point to your file
assert TOKEN and TOKEN != "PASTE_YOUR_BEARER_TOKEN_HERE", "Set a valid Bearer TOKEN."

page_csvs = process_pdf_multipage_to_csvs(
    pdf_path=MULTIPAGE_PDF,
    exemplars=exemplars,
    instructions=BASE_INSTRUCTIONS,
    max_workers=6,  # tune for your quota
)

# Save per-page CSV (only valid CSVs; errors into *_errors.txt)
base = pathlib.Path(MULTIPAGE_PDF).stem
errors_path = OUTPUT_DIR / f"{base}_errors.txt"
with errors_path.open("w", encoding="utf-8") as errf:
    for idx, csv_text in page_csvs.items():
        if csv_text.startswith("__ERROR__"):
            errf.write(f"Page {idx}: {csv_text}\n")
            continue
        out = OUTPUT_DIR / f"{base}_page{idx:03d}.csv"
        out.write_text(csv_text, encoding="utf-8")

# Combine and save to Excel
df_all = combine_page_csvs_to_dataframe(page_csvs, add_page_col=True)
display(df_all.head())

xlsx_out = OUTPUT_DIR / f"{base}_extract.xlsx"
with pd.ExcelWriter(xlsx_out, engine="openpyxl", mode="w") as xw:
    df_all.to_excel(xw, sheet_name="RENT_ROLL_EXTRACT", index=False)

xlsx_out, errors_path

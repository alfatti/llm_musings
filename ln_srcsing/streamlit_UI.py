"""
VLM Multi-Page PDF Extractor + Streamlit UI (OpenAI-style)

Requirements:
  pip install streamlit openai pdf2image pillow pandas

You also need poppler installed for pdf2image (platform-dependent).
"""

import os
import json
import base64
from io import BytesIO
from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile

import pandas as pd
import streamlit as st
from openai import OpenAI
from pdf2image import convert_from_path


# ============================================================
# Backend config (all creds come from environment, not UI)
# ============================================================

API_KEY = os.environ.get("OPENAI_API_KEY")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
USECASE_ID = os.environ.get("USECASE_ID", "VLM-EXTRACT")
APP_ID = os.environ.get("APP_ID", "rentroll-extractor-ui")
MODEL_NAME = os.environ.get("VLM_MODEL_NAME", "gpt-4.1-mini")  # or gemini proxy
DEFAULT_DPI = int(os.environ.get("VLM_PDF_DPI", "350"))

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)


# ============================================================
# Prompt + call helpers (OpenAI-style, generic but usable)
# ============================================================

def build_prompt(
    variable_name: str,
    section_variants: List[str],
    subsection_variants: List[str],
    line_item_variants: List[str],
    cell_hint: str,
    vicinity_rows: int,
    vicinity_cols: int,
) -> Tuple[str, Dict[str, Any]]:
    """
    Build (system_text, user_payload) for the vision call.
    user_payload is JSON-like instructions that we pass as text.
    """
    system_text = (
        "You are a meticulous financial document extraction assistant. "
        "You look at a page that is typically part of a loan report or rent roll. "
        "Your job is to find a specific variable and return it as structured JSON."
    )

    payload = {
        "task": f"Extract the variable: '{variable_name}'.",
        "name_pointers": {
            "section_variants": section_variants,
            "subsection_variants": subsection_variants,
            "line_item_variants": line_item_variants,
            "notes": (
                "Use fuzzy matching on section/sub-section/line-item labels, "
                "and rely on layout cues (proximity, row/column alignment)."
            ),
        },
        "cell_hint_and_vicinity": {
            "cell_hint": cell_hint,
            "vicinity_rows": vicinity_rows,
            "vicinity_cols": vicinity_cols,
            "notes": (
                "This hint indicates approximately where the value might be. "
                "Bias your search around this cell, then expand if needed."
            ),
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "variable_name": {"type": "string"},
                "value_raw": {"type": "string"},
                "unit": {"type": "string"},
                "cell_address": {"type": "string"},
                "confidence": {"type": "number"},
                "notes": {"type": "string"},
            },
            "required": ["variable_name", "value_raw", "confidence"],
            "additionalProperties": True,
        },
        "constraints": [
            "Return a SINGLE JSON object.",
            "Do not include any explanatory text outside the JSON.",
            "Return numeric values exactly as seen (including commas and signs).",
        ],
    }
    return system_text, payload


def assemble_headers(
    api_key: str,
    usecase_id: str,
    app_id: str,
) -> Dict[str, str]:
    """
    Optional headers for a proxy / gateway (if you use one).
    Here we just return metadata; OpenAI client itself handles auth.
    """
    return {
        "x-usecase-id": usecase_id,
        "x-app-id": app_id,
        # No Authorization header here; OpenAI client does that.
    }


def call_vlm_openai_style(
    openai_client: OpenAI,
    model: str,
    data_url: str,
    system_text: str,
    user_payload: Dict[str, Any],
    headers: Dict[str, str],
    temperature: float = 0.0,
    max_tokens: int = 800,
) -> Tuple[str, Dict[str, Any]]:
    """
    Call a vision-capable model using the OpenAI 'responses' API.
    data_url is a 'data:image/jpeg;base64,...' string.
    """
    user_text = (
        "Use the attached page image and the following JSON instructions. "
        "Return ONLY a JSON object that matches the 'output_schema' field.\n\n"
        "Instructions:\n"
        + json.dumps(user_payload, indent=2)
    )

    resp = openai_client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": [{"type": "text", "text": system_text}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_text},
                    {
                        "type": "input_image",
                        "image_url": {"url": data_url},
                    },
                ],
            },
        ],
        temperature=temperature,
        max_output_tokens=max_tokens,
        extra_headers=headers,
        response_format={"type": "json_object"},
    )

    # Extract text from the response object
    # (structure may vary slightly across model families)
    raw_text = resp.output[0].content[0].text
    return raw_text, resp.to_dict()


def coerce_json_from_text(raw_text: str) -> Any:
    """
    Try to parse model output as JSON, stripping code fences if needed.
    """
    txt = raw_text.strip()

    # Strip ```json ... ``` or ``` ... ```
    if txt.startswith("```"):
        # Remove opening fence
        first_newline = txt.find("\n")
        if first_newline != -1:
            txt = txt[first_newline + 1:]
        # Remove closing fence
        if txt.endswith("```"):
            txt = txt[:-3].strip()

    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        # As a last resort, try to salvage up to first/last brace
        start = txt.find("{")
        end = txt.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = txt[start : end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Could not parse JSON from model output:\n{raw_text}")


# ============================================================
# PDF -> image (data URLs)
# ============================================================

def pdf_to_data_urls(pdf_path: str, dpi: int = 350, jpeg_quality: int = 92) -> List[str]:
    """
    Convert a multi-page PDF to a list of JPEG data URLs (one per page).
    """
    pages = convert_from_path(pdf_path, dpi=dpi)
    if not pages:
        raise ValueError("No pages found in PDF.")

    data_urls: List[str] = []
    for page in pages:
        buf = BytesIO()
        page.convert("RGB").save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        data_urls.append("data:image/jpeg;base64," + b64)
    return data_urls


def pdf_to_data_url(pdf_path: str, dpi: int = 350, jpeg_quality: int = 92) -> str:
    """
    Backwards-compatible single-page PDF -> single data URL.
    Returns first page only.
    """
    return pdf_to_data_urls(pdf_path, dpi=dpi, jpeg_quality=jpeg_quality)[0]


# ============================================================
# Extraction functions (per data_url + wrapper)
# ============================================================

def run_extraction_for_data_url(
    openai_client: Any,
    data_url: str,
    variable_name: str,
    section_variants: List[str],
    subsection_variants: List[str],
    line_item_variants: List[str],
    cell_hint: str,
    vicinity_rows: int,
    vicinity_cols: int,
    api_key: str,
    usecase_id: str,
    app_id: str,
    model: str = MODEL_NAME,
) -> Dict[str, Any]:
    """
    Core extraction for a single page image (data_url).
    """
    system_text, user_payload = build_prompt(
        variable_name=variable_name,
        section_variants=section_variants,
        subsection_variants=subsection_variants,
        line_item_variants=line_item_variants,
        cell_hint=cell_hint,
        vicinity_rows=vicinity_rows,
        vicinity_cols=vicinity_cols,
    )

    headers = assemble_headers(api_key=api_key, usecase_id=usecase_id, app_id=app_id)

    raw_text, raw_resp = call_vlm_openai_style(
        openai_client=openai_client,
        model=model,
        data_url=data_url,
        system_text=system_text,
        user_payload=user_payload,
        headers=headers,
        temperature=0.0,
        max_tokens=800,
    )

    parsed = coerce_json_from_text(raw_text)

    return {
        "raw_text": raw_text,
        "parsed": parsed,
        "request_headers_used": headers,
        "raw_response": raw_resp,
    }


def run_extraction(
    openai_client: Any,
    pdf_path: str,
    variable_name: str,
    section_variants: List[str],
    subsection_variants: List[str],
    line_item_variants: List[str],
    cell_hint: str,
    vicinity_rows: int,
    vicinity_cols: int,
    dpi: int,
    api_key: str,
    usecase_id: str,
    app_id: str,
    model: str = MODEL_NAME,
) -> Dict[str, Any]:
    """
    Backwards-compatible single-page wrapper.
    """
    data_url = pdf_to_data_url(pdf_path, dpi=dpi)
    return run_extraction_for_data_url(
        openai_client=openai_client,
        data_url=data_url,
        variable_name=variable_name,
        section_variants=section_variants,
        subsection_variants=subsection_variants,
        line_item_variants=line_item_variants,
        cell_hint=cell_hint,
        vicinity_rows=vicinity_rows,
        vicinity_cols=vicinity_cols,
        api_key=api_key,
        usecase_id=usecase_id,
        app_id=app_id,
        model=model,
    )


# ============================================================
# Multi-page orchestration
# ============================================================

def extract_all_pages_from_pdf_bytes(
    openai_client: Any,
    pdf_bytes: bytes,
    variable_name: str,
    section_variants: List[str],
    subsection_variants: List[str],
    line_item_variants: List[str],
    cell_hint: str,
    vicinity_rows: int,
    vicinity_cols: int,
    dpi: int,
) -> Tuple[List[Tuple[int, Dict[str, Any]]], List[Tuple[int, str]]]:
    """
    Save uploaded PDF bytes to temp file, shred into page data URLs,
    and run extraction on each page in parallel.

    Returns:
      - results:  [(page_no, result_dict)]
      - failures: [(page_no, error_message)]
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    data_urls = pdf_to_data_urls(tmp_path, dpi=dpi)
    total_pages = len(data_urls)

    results: List[Tuple[int, Dict[str, Any]]] = []
    failures: List[Tuple[int, str]] = []

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(
                run_extraction_for_data_url,
                openai_client,
                data_url,
                variable_name,
                section_variants,
                subsection_variants,
                line_item_variants,
                cell_hint,
                vicinity_rows,
                vicinity_cols,
                API_KEY,
                USECASE_ID,
                APP_ID,
                MODEL_NAME,
            ): page_idx + 1
            for page_idx, data_url in enumerate(data_urls)
        }

        progress_bar = st.progress(0.0)
        completed = 0

        for fut in as_completed(futures):
            page_no = futures[fut]
            try:
                res = fut.result()
                results.append((page_no, res))
            except Exception as e:
                failures.append((page_no, str(e)))
            finally:
                completed += 1
                progress_bar.progress(completed / total_pages)

    results.sort(key=lambda x: x[0])
    return results, failures


def results_to_dataframe(
    results: List[Tuple[int, Dict[str, Any]]]
) -> Tuple[pd.DataFrame, List[Tuple[int, str]]]:
    """
    Convert [(page_no, result_dict), ...] into a single DataFrame.

    Assumes result_dict["parsed"] is a dict or list of dicts.
    Skips pages that fail conversion and logs them.
    """
    rows: List[Dict[str, Any]] = []
    skipped: List[Tuple[int, str]] = []

    for page_no, res in results:
        parsed = res.get("parsed")

        try:
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        row = {"page": page_no, **item}
                        rows.append(row)
                    else:
                        raise ValueError("List item is not a dict")
            elif isinstance(parsed, dict):
                row = {"page": page_no, **parsed}
                rows.append(row)
            else:
                raise ValueError("Parsed output is neither dict nor list")
        except Exception as e:
            skipped.append((page_no, f"Conversion to row failed: {e!r}"))

    if rows:
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame()

    return df, skipped


def parse_variants(text: str) -> List[str]:
    """
    Parse textarea input into list of strings (comma- or newline-separated).
    """
    if not text:
        return []
    pieces: List[str] = []
    for line in text.splitlines():
        for part in line.split(","):
            part = part.strip()
            if part:
                pieces.append(part)
    return pieces


# ============================================================
# Streamlit UI
# ============================================================

def main():
    st.set_page_config(
        page_title="VLM Multi-Page PDF Extractor",
        layout="wide",
    )

    st.title("📄 VLM Multi-Page PDF Extractor")
    st.markdown(
        "Upload a **multi-page PDF**. The app will shred it into pages, "
        "run the vision extractor on each page in parallel, combine the JSON outputs "
        "into a single table, and let you download a CSV."
    )

    if not API_KEY:
        st.error("Backend API key (OPENAI_API_KEY) is not set in the environment.")
        return

    # ---- Sidebar: extraction parameters (NO creds here) ----
    st.sidebar.header("📌 Extraction Parameters")

    variable_name = st.sidebar.text_input(
        "Variable Name",
        value="Outstanding Loan Balance",
    )

    section_text = st.sidebar.text_area(
        "Section Variants (one per line or comma-separated)",
        value="Loan Summary\nLoan Overview",
        height=80,
    )
    subsection_text = st.sidebar.text_area(
        "Sub-Section Variants",
        value="Current Balance\nOutstanding Balance",
        height=80,
    )
    line_item_text = st.sidebar.text_area(
        "Line-Item Variants",
        value="Outstanding Loan Balance\nOLB",
        height=80,
    )

    cell_hint = st.sidebar.text_input(
        "Cell Hint (optional)",
        value="N23",
        help="Example: 'N23'. Leave blank if not applicable.",
    )

    vicinity_rows = st.sidebar.number_input(
        "Vicinity Rows",
        min_value=1,
        max_value=30,
        value=8,
        step=1,
        help="Number of rows above/below the hint to scan.",
    )
    vicinity_cols = st.sidebar.number_input(
        "Vicinity Columns",
        min_value=1,
        max_value=30,
        value=8,
        step=1,
        help="Number of columns left/right of the hint to scan.",
    )

    dpi = st.sidebar.number_input(
        "PDF Rasterization DPI",
        min_value=100,
        max_value=600,
        value=DEFAULT_DPI,
        step=50,
    )

    section_variants = parse_variants(section_text)
    subsection_variants = parse_variants(subsection_text)
    line_item_variants = parse_variants(line_item_text)

    # ---- Main area: upload + run ----
    st.markdown("---")
    uploaded_pdf = st.file_uploader("Upload a multi-page PDF", type=["pdf"])

    run_button = st.button("🚀 Run Extraction", disabled=uploaded_pdf is None)

    if uploaded_pdf is None:
        st.info("⬆️ Upload a PDF to get started.")
        return

    if run_button:
        pdf_bytes = uploaded_pdf.read()

        st.write("Starting parallel extraction across pages...")
        with st.spinner("Shredding PDF into pages and calling the VLM backend..."):
            results, page_failures = extract_all_pages_from_pdf_bytes(
                openai_client=client,
                pdf_bytes=pdf_bytes,
                variable_name=variable_name,
                section_variants=section_variants,
                subsection_variants=subsection_variants,
                line_item_variants=line_item_variants,
                cell_hint=cell_hint,
                vicinity_rows=int(vicinity_rows),
                vicinity_cols=int(vicinity_cols),
                dpi=int(dpi),
            )

        if not results and not page_failures:
            st.error("No pages processed. Check the PDF and parameters.")
            return

        st.success(f"Extraction complete for {len(results)} page(s).")

        # Combine into DataFrame
        df, conversion_skips = results_to_dataframe(results)

        if df.empty:
            st.warning("Extraction returned no tabular data (empty DataFrame).")
        else:
            st.subheader("📊 Combined Extraction Results")
            st.dataframe(df, use_container_width=True)

            st.markdown("---")
            download_toggle = st.toggle("Enable CSV download")

            if download_toggle:
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv_bytes,
                    file_name="extraction_results.csv",
                    mime="text/csv",
                )

        # Report skipped pages
        skipped_msgs: List[str] = []

        if page_failures:
            for page_no, msg in page_failures:
                skipped_msgs.append(f"Page {page_no}: extraction failed ({msg})")

        if conversion_skips:
            for page_no, msg in conversion_skips:
                skipped_msgs.append(f"Page {page_no}: skipped during DataFrame conversion ({msg})")

        if skipped_msgs:
            st.markdown("---")
            st.subheader("⚠️ Pages Skipped / Errors")
            for line in skipped_msgs:
                st.write("• " + line)
        else:
            st.markdown("---")
            st.info("✅ All pages processed and included in the DataFrame.")


if __name__ == "__main__":
    main()

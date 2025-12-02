# vlm_streamlit_multi_page_app.py

import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple

import streamlit as st
import pandas as pd
from openai import OpenAI

from VLM_extractor_from_cropped_openai_style import (
    pdf_to_data_urls,
    run_extraction_for_data_url,
)

# -----------------------------
# Helpers
# -----------------------------

def parse_variants(text: str) -> List[str]:
    """
    Parse a text area input into a list of strings.
    Accepts comma-separated or newline-separated.
    """
    if not text:
        return []
    # Split on newlines first, then on commas for each line
    pieces: List[str] = []
    for line in text.splitlines():
        for part in line.split(","):
            part = part.strip()
            if part:
                pieces.append(part)
    return pieces


def make_openai_client(api_key: str, base_url: str) -> OpenAI:
    """
    Simple OpenAI client wrapper. Adjust base_url if needed
    for your enterprise Gemini/OpenAI proxy.
    """
    return OpenAI(api_key=api_key, base_url=base_url)


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
    api_key: str,
    usecase_id: str,
    app_id: str,
    model: str,
) -> Tuple[List[Tuple[int, Dict[str, Any]]], List[Tuple[int, str]]]:
    """
    Save uploaded PDF bytes to a temp file, shred into page data URLs,
    run extraction on each page in parallel, and return:
      - results: list of (page_no, result_dict)
      - failures: list of (page_no, error_message)
    """
    # Save to a temp file so we can reuse your existing pdf_to_data_urls
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    data_urls = pdf_to_data_urls(tmp_path, dpi=dpi)
    total_pages = len(data_urls)

    results: List[Tuple[int, Dict[str, Any]]] = []
    failures: List[Tuple[int, str]] = []

    # Use a pool to process pages in parallel
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
                api_key,
                usecase_id,
                app_id,
                model,
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

    # Sort by page number to keep order
    results.sort(key=lambda x: x[0])

    return results, failures


def results_to_dataframe(
    results: List[Tuple[int, Dict[str, Any]]]
) -> Tuple[pd.DataFrame, List[Tuple[int, str]]]:
    """
    Convert the list of (page_no, result_dict) into a single DataFrame.
    If any page's 'parsed' output can't be handled, skip that page
    and record a message.
    """
    rows: List[Dict[str, Any]] = []
    skipped: List[Tuple[int, str]] = []

    for page_no, res in results:
        parsed = res.get("parsed", None)

        try:
            if isinstance(parsed, list):
                # multiple rows per page
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


# -----------------------------
# Streamlit UI
# -----------------------------

def main():
    st.set_page_config(
        page_title="VLM Multi-Page PDF Extractor",
        layout="wide",
    )

    st.title("📄 VLM Multi-Page PDF Extractor")

    st.markdown(
        "Upload a **multi-page PDF**, run extraction on **each page in parallel**, "
        "and download a combined CSV of all JSON outputs."
    )

    # --- Sidebar: config ---
    st.sidebar.header("🔑 API & Model Settings")

    api_key = st.sidebar.text_input("API Key", type="password")
    base_url = st.sidebar.text_input(
        "Base URL",
        value="https://api.openai.com/v1",
        help="Use your enterprise / proxy URL if applicable.",
    )
    model = st.sidebar.text_input("Model", value="gemini-2.5.pro")

    usecase_id = st.sidebar.text_input("Usecase ID", value="VLM-EXTRACT")
    app_id = st.sidebar.text_input("App ID", value="rentroll-extractor-ui")

    dpi = st.sidebar.number_input("PDF Rasterization DPI", min_value=100, max_value=600, value=350, step=50)

    st.sidebar.header("📌 Extraction Parameters")

    variable_name = st.sidebar.text_input("Variable Name", value="Outstanding Loan Balance")
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
        help="How many rows above/below the hint to scan.",
    )
    vicinity_cols = st.sidebar.number_input(
        "Vicinity Columns",
        min_value=1,
        max_value=30,
        value=8,
        step=1,
        help="How many columns left/right of the hint to scan.",
    )

    section_variants = parse_variants(section_text)
    subsection_variants = parse_variants(subsection_text)
    line_item_variants = parse_variants(line_item_text)

    # --- Main area: file upload & run ---

    st.markdown("---")
    uploaded_pdf = st.file_uploader("Upload a multi-page PDF", type=["pdf"])

    run_button = st.button("🚀 Run Extraction", disabled=uploaded_pdf is None or not api_key)

    if uploaded_pdf is None:
        st.info("⬆️ Upload a PDF to get started.")
        return

    if not api_key:
        st.warning("Enter your API key in the sidebar to enable extraction.")
        return

    if run_button:
        st.write("Initializing client and starting extraction...")
        client = make_openai_client(api_key=api_key, base_url=base_url)

        pdf_bytes = uploaded_pdf.read()

        with st.spinner("Shredding PDF into pages and extracting in parallel..."):
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
                api_key=api_key,
                usecase_id=usecase_id,
                app_id=app_id,
                model=model,
            )

        if not results and not page_failures:
            st.error("No pages processed. Check the PDF and parameters.")
            return

        st.success(f"Extraction complete for {len(results)} page(s).")

        # Convert to DataFrame
        df, conversion_skips = results_to_dataframe(results)

        if df.empty:
            st.warning("Extraction returned no tabular data (empty DataFrame).")
        else:
            st.subheader("📊 Combined Extraction Results")
            st.dataframe(df, use_container_width=True)

            # --- Download toggle ---
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

        # --- Reporting skipped pages ---
        skipped_msgs = []

        if page_failures:
            for page_no, msg in page_failures:
                skipped_msgs.append(f"Page {page_no}: extraction failed ({msg})")
        if conversion_skips:
            for page_no, msg in conversion_skips:
                skipped_msgs.append(f"Page {page_no}: skipped in DataFrame conversion ({msg})")

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

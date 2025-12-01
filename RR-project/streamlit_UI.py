# app_rentroll_spreader.py

import io
import pandas as pd
import streamlit as st

# --- IMPORT YOUR EXISTING MODULE HERE ---
# from your_module_name import spread_rentroll


# --- OPTIONAL: dummy placeholder for spread_rentroll for testing ---
def spread_rentroll(file_bytes: bytes, asset_type: str) -> pd.DataFrame:
    """
    Dummy implementation for testing the UI.
    Replace this with: from your_module_name import spread_rentroll
    and remove this function.
    """
    # Just return a sample DF to prove the UI works
    return pd.DataFrame(
        {
            "Unit": ["101", "102", "103"],
            "Tenant": ["Alice LLC", "Bob Inc.", "Charlie Co."],
            "Rent": [2500, 2700, 3000],
            "Asset_Type": [asset_type] * 3,
        }
    )


def main():
    st.set_page_config(
        page_title="Rent Roll Spreader",
        layout="wide",
    )

    # --- Sidebar: Company Logo Placeholder ---
    with st.sidebar:
        st.markdown("### Company")
        # Replace "company_logo.png" with your actual logo file path
        st.image("company_logo.png", caption="Your Company Name", use_column_width=True)

        st.markdown("---")
        st.markdown("Rent Roll Spreading UI")

    st.title("Rent Roll Spreader")
    st.write("Upload a rent roll PDF and spread it into a structured CSV.")

    # --- Asset Type Selection ---
    asset_type_label = st.radio(
        "Asset type of the rent roll:",
        options=["Multi-family", "Commercial"],
        index=0,
        horizontal=True,
    )

    # Map UI label to what spread_rentroll expects
    asset_type_map = {
        "Multi-family": "multifamily",
        "Commercial": "commercial",
    }
    asset_type = asset_type_map[asset_type_label]

    # --- File Uploader ---
    uploaded_file = st.file_uploader(
        "Upload rent roll PDF",
        type=["pdf"],
        help="Drag and drop or browse to select a PDF rent roll.",
    )

    # Placeholder for status messages and results
    status_placeholder = st.empty()
    df_placeholder = st.empty()
    download_placeholder = st.empty()

    # --- Process Button ---
    process_button = st.button(
        "Process Rent Roll",
        type="primary",
        disabled=uploaded_file is None,
    )

    if process_button:
        if uploaded_file is None:
            status_placeholder.warning("Please upload a rent roll PDF before processing.")
            return

        # Show processing indicator
        with st.spinner("Spreading rent roll… please wait."):
            status_placeholder.info("Processing the uploaded rent roll…")

            # Read file bytes
            file_bytes = uploaded_file.read()

            # Call your existing spread_rentroll() module
            # Example signature: spread_rentroll(file_bytes, asset_type)
            df = spread_rentroll(file_bytes=file_bytes, asset_type=asset_type)

        status_placeholder.success("Processing complete! Preview and download your CSV below.")
        df_placeholder.dataframe(df, use_container_width=True)

        # Convert to CSV for download
        csv_bytes = df.to_csv(index=False).encode("utf-8")

        download_placeholder.download_button(
            label="Download CSV",
            data=csv_bytes,
            file_name="spread_rentroll_output.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()

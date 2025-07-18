# Design Choices: Split-screen upload, accordion for instructions, centralized titles, emoji icons, clean message flow.
# User Feedback: Uses st.info, st.success, and st.error for responsive messages.
# Assumptions:
# your_pipeline.py has extract_rent_roll(df) and enrich_with_concessions(df, concessions) pre-written.
# Replace "your_logo.png" with the path to your company logo.
# Downloads: .to_excel(index=False) outputs ready-to-save file via Streamlit.

import streamlit as st
import pandas as pd
from io import BytesIO

# === Importing your modules ===
from rent_roll_extractor import extract_rent_roll
from concession_joiner import join_concessions

# === Page Config ===
st.set_page_config(
    page_title="RR AI Extractor",
    page_icon="🏢",
    layout="centered"
)

# === Header ===
# === Header ===
col_logo, col_title = st.columns([1, 5])

with col_logo:
    st.image("company_logo.png", width=80)  # Adjust size as needed

with col_title:
    st.markdown("<h1 style='margin-bottom: 0;'>RR AI Extractor</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='margin-top: 0;'>AI-powered Rent Roll Data Extraction Tool</h4>", unsafe_allow_html=True)

st.markdown("---")


# === Logo Placeholder ===
st.image("your_logo.png", width=200)  # Replace with actual logo path or remove

# === Instructions ===
with st.expander("ℹ️ Instructions (Click to expand)"):
    st.markdown("""
    **Before uploading your sheets, please ensure:**
    - The file is in plain **tabular format**
    - Irregular headers (like titles, unit type rows) are removed
    - Rows with summary statistics (e.g., totals) are removed
    """)

st.markdown("## 📤 Upload Files")

# === Upload Fields ===
col1, col2 = st.columns(2)

with col1:
    rent_roll_file = st.file_uploader("📄 Upload Rent Roll Excel", type=["xlsx", "xls"], key="rent_roll")

with col2:
    concession_file = st.file_uploader("📄 Upload Concession Sheet (Optional)", type=["xlsx", "xls"], key="concession")

# === Abort Button ===
if st.button("❌ Abort / Clear"):
    st.session_state.clear()
    st.rerun()

# === START BUTTON ===
st.markdown("###")
start_extraction = st.button("▶️ Start Extraction", disabled=not rent_roll_file)

# === Extraction Logic ===
if start_extraction:
    if rent_roll_file and not concession_file:
        st.info("📢 Only Rent Roll uploaded. Proceeding with extraction only...")

        with st.spinner("🔄 Running extraction..."):
            rent_df = pd.read_excel(rent_roll_file)
            extracted_df = extract_rent_roll(rent_df)

        st.success("✅ Extraction complete!")
        buffer = BytesIO()
        extracted_df.to_excel(buffer, index=False)
        st.download_button(
            label="📥 Download Extracted Rent Roll",
            data=buffer.getvalue(),
            file_name="rent_roll_extract.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    elif rent_roll_file and concession_file:
        st.info("📢 Both Rent Roll and Concession uploaded. Proceeding with full enrichment...")

        with st.spinner("🔄 Running extraction and joining..."):
            rent_df = pd.read_excel(rent_roll_file)
            cons_df = pd.read_excel(concession_file)
            extracted_df = extract_rent_roll(rent_df)
            final_df = join_concessions(extracted_df, cons_df)

        st.success("✅ Extraction and enrichment complete!")
        buffer = BytesIO()
        final_df.to_excel(buffer, index=False)
        st.download_button(
            label="📥 Download Final Joined File",
            data=buffer.getvalue(),
            file_name="rent_roll_final.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    elif concession_file and not rent_roll_file:
        st.error("⚠️ Error: Rent Roll is required to proceed.")
# === Footer ===
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 0.85em;'>Built for internal demo. © 2025</p>", unsafe_allow_html=True)

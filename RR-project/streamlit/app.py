# Design Choices: Split-screen upload, accordion for instructions, centralized titles, emoji icons, clean message flow.
# User Feedback: Uses st.info, st.success, and st.error for responsive messages.
# Assumptions:
# your_pipeline.py has extract_rent_roll(df) and enrich_with_concessions(df, concessions) pre-written.
# Replace "your_logo.png" with the path to your company logo.
# Downloads: .to_excel(index=False) outputs ready-to-save file via Streamlit.

import streamlit as st
import pandas as pd

# Assume these functions are pre-written
from your_pipeline import extract_rent_roll, enrich_with_concessions

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="RR AI Extractor",
    page_icon="🏢",
    layout="centered"
)

# ---------- HEADER / LOGO ----------
st.markdown("<h1 style='text-align: center;'>🏢 RR AI Extractor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>AI-powered Rent Roll Data Extraction Tool</h4>", unsafe_allow_html=True)
st.markdown("---")

# ---------- LOGO PLACEHOLDER ----------
st.image("your_logo.png", width=200)  # Replace with actual logo path or URL

# ---------- INSTRUCTIONS ----------
with st.expander("ℹ️ Instructions (Click to expand)"):
    st.markdown("""
    **Before uploading your sheets, please ensure:**
    - The file is in plain **tabular format**
    - Irregular headers (like titles, unit type rows) are removed
    - Rows with summary statistics (e.g., totals) are removed
    """)

st.markdown("## 📤 Upload Files")

# ---------- FILE UPLOADS ----------
col1, col2 = st.columns(2)

with col1:
    rent_roll_file = st.file_uploader("📄 Upload Rent Roll Excel", type=["xlsx", "xls"], key="rent_roll")

with col2:
    concession_file = st.file_uploader("📄 Upload Concession Sheet (Optional)", type=["xlsx", "xls"], key="concession")

# ---------- PROCESSING LOGIC ----------
if rent_roll_file and not concession_file:
    st.info("📢 Only Rent Roll uploaded. Proceeding with extraction only...")
    rent_df = pd.read_excel(rent_roll_file)
    extracted_df = extract_rent_roll(rent_df)
    st.success("✅ Extraction complete!")
    st.download_button("📥 Download Extracted File", data=extracted_df.to_excel(index=False), file_name="rent_roll_extract.xlsx")

elif rent_roll_file and concession_file:
    st.info("📢 Both Rent Roll and Concession uploaded. Proceeding with full enrichment...")
    rent_df = pd.read_excel(rent_roll_file)
    cons_df = pd.read_excel(concession_file)
    extracted_df = extract_rent_roll(rent_df)
    final_df = enrich_with_concessions(extracted_df, cons_df)
    st.success("✅ Extraction and joining complete!")
    st.download_button("📥 Download Final Extract", data=final_df.to_excel(index=False), file_name="rent_roll_final.xlsx")

elif concession_file and not rent_roll_file:
    st.error("⚠️ Error: Rent Roll is required to proceed.")

else:
    st.warning("Please upload at least the Rent Roll file.")

# ---------- FOOTER ----------
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 0.85em;'>Built for internal demo. © 2025</p>", unsafe_allow_html=True)

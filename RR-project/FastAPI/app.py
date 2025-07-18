import streamlit as st
import requests
import pandas as pd
from io import BytesIO

# --- Configuration ---
FASTAPI_BASE_URL = "http://localhost:8000"  # Change to deployed URL if needed

st.set_page_config(
    page_title="RR AI Extractor",
    page_icon="company_logo.png",
    layout="centered"
)

# --- Header ---
col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.image("company_logo.png", width=80)
with col_title:
    st.markdown("<h1 style='margin-bottom: 0;'>RR AI Extractor</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='margin-top: 0;'>AI-powered Rent Roll Data Extraction Tool</h4>", unsafe_allow_html=True)
st.markdown("---")

# --- Instructions ---
with st.expander("ℹ️ Instructions (Click to expand)"):
    st.markdown("""
    **Before uploading your sheets, please ensure:**
    - The file is in plain **tabular format**
    - Irregular headers (like titles, unit type rows) are removed
    - Rows with summary statistics (e.g., totals) are removed
    """)

st.markdown("## 📤 Upload Files")

# --- Upload Widgets ---
col1, col2 = st.columns(2)
with col1:
    rent_roll_file = st.file_uploader("📄 Upload Rent Roll Excel", type=["xlsx", "xls"], key="rent_roll")
with col2:
    concession_file = st.file_uploader("📄 Upload Concession Sheet (Optional)", type=["xlsx", "xls"], key="concession")

# --- Submit Logic ---
if rent_roll_file and not concession_file:
    st.info("📢 Only Rent Roll uploaded. Proceeding with extraction only...")
    if st.button("🔍 Run Extraction"):
        files = {'rent_roll': rent_roll_file}
        response = requests.post(f"{FASTAPI_BASE_URL}/extract", files=files)
        if response.status_code == 200:
            st.success("✅ Extraction complete!")
            st.download_button(
                label="📥 Download Extracted Rent Roll",
                data=response.content,
                file_name="rent_roll_extract.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.error(f"❌ Error: {response.status_code} - {response.text}")

elif rent_roll_file and concession_file:
    st.info("📢 Both Rent Roll and Concession uploaded. Proceeding with full enrichment...")
    if st.button("🔍 Run Extraction + Join"):
        files = {
            'rent_roll': rent_roll_file,
            'concession': concession_file
        }
        response = requests.post(f"{FASTAPI_BASE_URL}/extract_and_join", files=files)
        if response.status_code == 200:
            st.success("✅ Extraction and joining complete!")
            st.download_button(
                label="📥 Download Final Extract",
                data=response.content,
                file_name="rent_roll_final.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.error(f"❌ Error: {response.status_code} - {response.text}")

elif concession_file and not rent_roll_file:
    st.error("⚠️ Error: Rent Roll is required to proceed.")

else:
    st.warning("Please upload at least the Rent Roll file.")

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 0.85em;'>Built for internal demo. © 2025</p>", unsafe_allow_html=True)

import pdfplumber
import pandas as pd
import streamlit as st
from google.cloud import aiplatform
from vertexai.preview.language_models import GenerativeModel
import json
import tempfile
import os

# --- Streamlit App Title ---
st.set_page_config(page_title="Rent Roll Extraction Tool", layout="centered")
st.title("🏢 Rent Roll Extraction Tool")
st.markdown("Upload a rent roll PDF, and we'll extract it into a spreadsheet.")

# --- Logo Placeholder ---
st.markdown("![Company Logo](https://via.placeholder.com/150x50?text=Logo+Here)")

# --- File Upload ---
uploaded_file = st.file_uploader("Choose a Rent Roll PDF", type="pdf")

# --- Feedback storage file ---
FEEDBACK_LOG = "feedback_log.csv"

# --- Helper Functions ---
def extract_pdf_text(upload):
    with pdfplumber.open(upload) as pdf:
        return "\n\n".join(page.extract_text() for page in pdf if page.extract_text())

def build_prompt(text):
    return f"""
You are a data extraction engine. Your task is to convert rent roll data in a semi-structured format into a normalized JSON list.

From the following rent roll text, extract one JSON object per charge per unit per location. Each object should have these fields:

- account_name
- as_of_date
- location
- unit
- tenant
- charge_date
- charge_type
- monthly_amount
- gpr
- dollar_per_sqft
- non_gpr
- gla_sqft
- lease_expiration
- move_in
- move_out

Also include unit totals or property totals as separate objects where applicable.

Here is the rent roll data:
""" 
{text}
"""
Return the result as a valid JSON array.
"""

def call_gemini(prompt, project_id):
    aiplatform.init(project=project_id, location="us-central1")
    model = GenerativeModel("gemini-1.5-pro-preview-0409")
    response = model.generate_content(prompt)
    return response.text

def save_feedback(feedback):
    df = pd.DataFrame([feedback])
    if os.path.exists(FEEDBACK_LOG):
        df.to_csv(FEEDBACK_LOG, mode='a', header=False, index=False)
    else:
        df.to_csv(FEEDBACK_LOG, index=False)

# --- Main Logic ---
if uploaded_file:
    with st.spinner("Extracting text from PDF..."):
        pdf_text = extract_pdf_text(uploaded_file)
        prompt = build_prompt(pdf_text)

    with st.spinner("Calling Gemini to extract structured data..."):
        try:
            raw_json = call_gemini(prompt, project_id="your-gcp-project-id")
            data = pd.read_json(raw_json)

            st.success("✅ Extraction complete!")
            st.subheader("Extracted Data Schema:")
            st.write(data.dtypes)

            st.subheader("Preview:")
            st.dataframe(data.head())

            tmp_download = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            data.to_csv(tmp_download.name, index=False)
            with open(tmp_download.name, "rb") as f:
                st.download_button(
                    label="📥 Download CSV",
                    data=f,
                    file_name="rent_roll_extracted.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error("Error parsing output from Gemini")
            st.text(raw_json)

    st.markdown("---")
    st.subheader("Was this output useful?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 Yes"):
            save_feedback({"file_name": uploaded_file.name, "feedback": "up"})
            st.success("Thanks for your feedback!")
    with col2:
        if st.button("👎 No"):
            save_feedback({"file_name": uploaded_file.name, "feedback": "down"})
            st.success("We'll use your feedback to improve.")

import pdfplumber
import pandas as pd
from google.cloud import aiplatform
from vertexai.preview.language_models import GenerativeModel

# --- 1. Load PDF and extract text ---
def extract_pdf_text(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return "\n\n".join(page.extract_text() for page in pdf if page.extract_text())

# --- 2. Construct the Gemini Prompt ---
def build_prompt(rent_roll_text):
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
\"\"\" 
{rent_roll_text}
\"\"\"
Return the result as a valid JSON array.
"""

# --- 3. Query Gemini 1.5 Pro on Vertex AI ---
def call_gemini(prompt, project_id, location="us-central1"):
    aiplatform.init(project=project_id, location=location)
    model = GenerativeModel("gemini-1.5-pro-preview-0409")
    response = model.generate_content(prompt)
    return response.text  # Expected to be a JSON string

# --- 4. Orchestrate ---
def main():
    project_id = "your-gcp-project-id"
    rent_pdf_path = "rent_roll_sample_final.pdf"

    text = extract_pdf_text(rent_pdf_path)
    prompt = build_prompt(text)
    raw_json = call_gemini(prompt, project_id)

    # Parse and display
    try:
        data = pd.read_json(raw_json)
        data.to_csv("rent_roll_output.csv", index=False)
        print("Extraction complete. Output saved to rent_roll_output.csv.")
    except Exception as e:
        print("Error parsing Gemini output:", e)
        print("Raw Output:\n", raw_json)

if __name__ == "__main__":
    main()


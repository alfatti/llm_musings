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
    return """
You are a data extraction engine. Your task is to convert a rent roll into a normalized JSON list, extracting all details at the most granular level (one row per charge per unit).

⚠️ STRICT INSTRUCTIONS:
- You must extract **every unit and every charge** from the rent roll.
- If a page contains *k* units, you must produce output for **all k units**.
- Do not skip vacant units or totals.
- All fields must be returned in every object. Use `null` if a value is missing.

For each charge line and each unit, extract the following 18 fields:

- account_name
- as_of_date
- rent_roll_page
- location
- unit
- market_rent
- current_rent
- tenant
- next_charge_date
- charge_type
- monthly_amount
- gpr
- dollar_per_sqft
- non_gpr
- gla_per_sqft
- lease_expiration
- move_in
- move_out

---

💡 Example input:

***
Land Lord Corporation 1

Rent Roll

As of Date: 10-31-2024
10844 Plaza

1 3,815.37 10844-Lowe’s;02-01-2025 CAM charge  
2,515.86 .00 .00 30,000.52 12-31-99 01-01-2020

02-01-2025 Management Fee  
1,000.00 .00 .00 9,000.00

01-31-2025 Miscellaneous charge  
200.00 .00 .00 3,100.00

1 Unit Totals:  
3,715.86 .00 .00 42,100.52

2 7,200.12 10844-ABC;02-01-2025 CAM charge  
250.00 .00 2.66 3,008.28 1,174.99 03-31-2035 04-04-2022

02-01-2025 Insurance charge  
30.05 .00 .36 426.60

02-01-2025 Property tax  
200.05 .00 2.35 2,696.16

02-01-2025 Rent charge  
3,500.00 40,200.00 39.34

02-01-2025 Trash charge  
62.72 .00 .50 805.80

2 Unit Totals:  
7,756.68 40,200.00 45.21 49,037.3

💡 Correct Output (abbreviated):

[
  {
    "account_name": "Land Lord Corporation",
    "as_of_date": "2024-10-31",
    "rent_roll_page": 1,
    "location": "10844 Plaza",
    "unit": "1",
    "market_rent": 3815.37,
    "current_rent": null,
    "tenant": "Lowe’s",
    "next_charge_date": "02-01-2025",
    "charge_type": "CAM charge",
    "monthly_amount": 2515.86,
    "gpr": 0.0,
    "dollar_per_sqft": 0.0,
    "non_gpr": 30000.52,
    "gla_per_sqft": null,
    "lease_expiration": "12-31-99",
    "move_in": "01-01-2020",
    "move_out": null
  },
  {
    "account_name": "Land Lord Corporation",
    "as_of_date": "2024-10-31",
    "rent_roll_page": 1,
    "location": "10844 Plaza",
    "unit": "2",
    "market_rent": 7200.12,
    "current_rent": null,
    "tenant": "ABC",
    "next_charge_date": "02-01-2025",
    "charge_type": "CAM charge",
    "monthly_amount": 250.00,
    "gpr": 0.0,
    "dollar_per_sqft": 2.66,
    "non_gpr": 3008.28,
    "gla_per_sqft": 1174.99,
    "lease_expiration": "03-31-2035",
    "move_in": "04-04-2022",
    "move_out": null
  }
]

---

Now extract and return a complete JSON array in this exact structure for the following rent roll:
\"\"\"
{{RENT_ROLL_TEXT}}
\"\"\"
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


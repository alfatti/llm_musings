import pdfplumber
import pandas as pd
import json
from typing import List
from google.cloud import aiplatform
from vertexai.preview.language_models import GenerativeModel

# ---- Prompt Template ----
GEMINI_PROMPT_TEMPLATE = """
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
- unit_total_monthly
- unit_total_gpr
- unit_total_dollar_per_sqft
- unit_total_non_gpr

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
    "unit_total_monthly": 3,715.86,
    "unit_total_gpr": 0.0,
    "unit_total_dollar_per_sqft": 0.0,
    "unit_total_non_gpr": 42,100.52
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
    "move_out": null,
    "unit_total_monthly": 7,756.68,
    "unit_total_gpr": 40,200.00,
    "unit_total_dollar_per_sqft": 45.21,
    "unit_total_non_gpr": 49,037.3
  }
]

---

Now extract and return a complete JSON array in this exact structure for the following rent roll (page {{PAGE_NUM}}):
\"\"\"
{{RENT_ROLL_TEXT}}
\"\"\"
"""

# ---- Helper Functions ----
def extract_pages(pdf_path: str) -> List[str]:
    with pdfplumber.open(pdf_path) as pdf:
        return [page.extract_text() for page in pdf if page.extract_text()]

def make_prompt(page_text: str, page_num: int) -> str:
    return GEMINI_PROMPT_TEMPLATE.replace("{{RENT_ROLL_TEXT}}", page_text).replace("{{PAGE_NUM}}", str(page_num))

def query_gemini(prompt: str, model: GenerativeModel) -> List[dict]:
    response = model.generate_content(prompt)
    try:
        return json.loads(response.text)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse Gemini output on one page: {e}")
        print("Raw Output:\n", response.text[:1000])
        return []

# ---- Main Function ----
def process_rent_roll(pdf_path: str, project_id: str, location: str = "us-central1"):
    aiplatform.init(project=project_id, location=location)
    model = GenerativeModel("gemini-1.5-pro-preview-0409")

    pages = extract_pages(pdf_path)
    all_results = []

    for i, page_text in enumerate(pages, start=1):
        print(f"🔍 Processing page {i}/{len(pages)}...")
        prompt = make_prompt(page_text, i)
        results = query_gemini(prompt, model)
        all_results.extend(results)

    df = pd.DataFrame(all_results)
    df.to_csv("rent_roll_output.csv", index=False)
    print("✅ Extraction complete. Saved to rent_roll_output.csv.")

# --- Usage ---
# process_rent_roll("rent_roll_sample_final.pdf", project_id="your-gcp-project-id")

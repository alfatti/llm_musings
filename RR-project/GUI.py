# Notebook-compatible version of Rent Roll GenAI Analysis
import pandas as pd
import pdfplumber
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

---

Now extract and return a complete JSON array in this exact structure for the following rent roll (page {{PAGE_NUM}}):
"""
{{RENT_ROLL_TEXT}}
"""

# ---- Helper Functions ----
def extract_pages(pdf_path) -> List[str]:
    with pdfplumber.open(pdf_path) as pdf:
        return [page.extract_text() for page in pdf if page.extract_text()]

def make_prompt(page_text: str, page_num: int) -> str:
    return GEMINI_PROMPT_TEMPLATE.replace("{{RENT_ROLL_TEXT}}", page_text).replace("{{PAGE_NUM}}", str(page_num))

def query_gemini(prompt: str, model: GenerativeModel) -> List[dict]:
    response = model.generate_content(prompt)
    try:
        return json.loads(response.text)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse Gemini output on page {page_num}: {e}")
        print("Raw Output Preview:\n", response.text[:1000])
        return []

# ---- Main Execution Block for Notebook ----
def process_rent_roll_notebook(pdf_path: str, project_id: str):
    aiplatform.init(project=project_id, location="us-central1")
    model = GenerativeModel("gemini-1.5-pro-preview-0409")

    pages = extract_pages(pdf_path)
    all_results = []

    for i, page_text in enumerate(pages, start=1):
        print(f"\n🔍 Processing page {i}...")
        prompt = make_prompt(page_text, i)
        results = query_gemini(prompt, model)
        if results:
            df = pd.DataFrame(results)
            for col in ['market_rent', 'current_rent', 'monthly_amount', 'gpr', 'dollar_per_sqft', 'non_gpr']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            display(df.style.format({
                'market_rent': '${:,.2f}',
                'current_rent': '${:,.2f}',
                'monthly_amount': '${:,.2f}',
                'gpr': '${:,.2f}',
                'dollar_per_sqft': '{:,.2f}',
                'non_gpr': '${:,.2f}'
            }))
            print("\nSchema:")
            display({col: str(dtype) for col, dtype in df.dtypes.items()})
            print("\nSummary Stats:")
            display(df.describe(include='all'))
            all_results.extend(results)

    if all_results:
        df_all = pd.DataFrame(all_results)
        df_all.to_csv("rent_roll_output.csv", index=False)
        print("\n✅ All pages processed. CSV saved as 'rent_roll_output.csv'")

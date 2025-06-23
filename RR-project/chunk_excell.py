import pandas as pd
import json
from google.cloud import aiplatform
from google.cloud.aiplatform.generation_models import GenerationModel
from google.auth import default

# --- CONFIGURATION ---
PROJECT_ID = "your-gcp-project-id"
LOCATION = "us-central1"
INPUT_FILE = "leases.xlsx"  # or "leases.csv"
OUTPUT_FILE = "parsed_leases.json"
MODEL = "gemini-1.5-pro-002"

# --- LOAD FILE ---
if INPUT_FILE.endswith(".csv"):
    df = pd.read_csv(INPUT_FILE)
else:
    df = pd.read_excel(INPUT_FILE)

table_string = df.to_markdown(index=False)

# --- CONSTRUCT PROMPT ---
prompt = f"""
You are an expert at extracting structured data from spreadsheet-style tables.

The table below contains multiple commercial lease records for different tenants. Each tenant's data is grouped in a contiguous block of rows and includes:

1. Lease Summary: general lease info like property name (with ID), unit(s), tenant name (with ID), lease dates, rent values, etc.
2. Rent Steps: rent changes over time
3. Charge Schedule: CAM and operating charges
4. Amendments: summary of changes and dates

Your task is to:
- Segment the table into chunks per tenant
- Extract:
    - Property Name and Property ID (e.g., "ABC Center (br1234)")
    - Tenant Name and Tenant ID (e.g., "Banana LLC (psb01234)")
- Structure each chunk into a dictionary with the following format:

{
  "Tenant Name": {
    "Tenant ID": "...",
    "Property": "...",
    "Property ID": "...",
    "Lease Summary": {{...}},
    "Rent Steps": [{{...}}, ...],
    "Charge Schedule": [{{...}}, ...],
    "Amendments": [{{...}}, ...]
  },
  ...
}

Make sure to:
- Extract all monetary values as numbers
- Format dates as YYYY-MM-DD
- Use `null` for any missing field
- Do not summarize or skip any tenant
- Do not wrap the JSON in markdown code blocks

Here is the input table:

<spreadsheet>
{table_string}
</spreadsheet>
"""

# --- INIT GENAI ---
aiplatform.init(project=PROJECT_ID, location=LOCATION)

model = GenerationModel(
    model_name=MODEL,
    generation_config={"temperature": 0.2, "max_output_tokens": 8192}
)

response = model.generate_content(prompt)

# --- EXTRACT & CLEAN ---
raw_output = response.candidates[0].content.parts[0].text.strip()

# Remove triple backticks if Gemini returns a markdown-formatted block
cleaned = raw_output
if cleaned.startswith("```json") or cleaned.startswith("```"):
    cleaned = cleaned.strip("` \n")
    lines = cleaned.splitlines()
    cleaned = "\n".join(line for line in lines if not line.strip().startswith("```"))

try:
    parsed = json.loads(cleaned)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(parsed, f, indent=2)
    print(f"✅ Parsed lease data saved to {OUTPUT_FILE}")
except json.JSONDecodeError as e:
    print("⚠️ Could not decode JSON. Here's the raw output:\n")
    print(raw_output)

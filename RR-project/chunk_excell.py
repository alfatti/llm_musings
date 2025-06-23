import pandas as pd
import json
from langchain_google_vertexai import ChatVertexAI

def extract_lease_blocks_from_table(
    df: pd.DataFrame,
    model_name: str = "gemini-1.5-pro-002",
    temperature: float = 0.2,
    max_output_tokens: int = 8192,
) -> dict:
    """
    Sends a lease table to Gemini Pro 2.5 via LangChain, and returns structured lease data.
    """

    table_string = df.to_markdown(index=False)

    prompt_template = """
You are an expert at parsing commercial lease tables.

The spreadsheet below contains multiple tenants, where each tenant’s data is grouped in a block of rows. Each tenant block includes:

1. Lease Summary: general info like Property, Unit(s), Tenant, Area, Lease From/To, Monthly Rent, etc.
2. Rent Steps: rows showing rent changes over time
3. Charge Schedule: rows showing CAM, utilities, insurance, etc.
4. Amendments: lease status changes, activation dates, notes

---

Your task is to:
- Identify each tenant’s data block
- Extract:
    - Tenant Name (e.g. "Banana Partners & Associates, LLC")
    - Tenant ID (from parentheses, e.g. psb01234)
    - Property Name (e.g. "ABC Center 18")
    - Property ID (from parentheses, e.g. br1234)
- Output each tenant as an entry in a top-level JSON dictionary

Each tenant’s data must be formatted as:

- Key: Tenant Name (string)
- Value: dictionary with the following keys:
    - "Tenant ID": string
    - "Property": string
    - "Property ID": string
    - "Lease Summary": dictionary of lease metadata
    - "Rent Steps": list of dictionaries
    - "Charge Schedule": list of dictionaries
    - "Amendments": list of dictionaries

---

🛑 JSON Format Requirements:
- Output must be valid JSON
- DO NOT use triple backticks (```) or markdown
- All keys and strings must use double quotes ("")
- If any value is missing, set it to null
- DO NOT skip any of the required keys
- DO NOT include explanations or extra text — only return the JSON

---

Here is the input spreadsheet:

<spreadsheet>
{}</spreadsheet>
""".format(table_string)

    # Initialize Gemini via LangChain
    model = ChatVertexAI(
        model_name=model_name,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )

    response = model.invoke(prompt_template)
    raw_output = response.content.strip()

    # Strip markdown blocks if Gemini adds them anyway
    if raw_output.startswith("```"):
        raw_output = raw_output.strip("` \n")
        raw_output = "\n".join(
            line for line in raw_output.splitlines() if not line.strip().startswith("```")
        )

    try:
        return json.loads(raw_output)
    except json.JSONDecodeError as e:
        raise ValueError(f"❌ Gemini returned malformed JSON:\n\n{raw_output}") from e

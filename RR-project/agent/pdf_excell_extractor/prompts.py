### file: prompts.py
# ----- Extraction prompts (slightly different for PDF vs Excel layouts) -----

EXTRACTOR_PROMPT_PDF = """
You are an expert rent‑roll extractor for PDFs. Given the following page text,
return **ALL** units and charges in strict JSON array format:
[
  {
    "unit": "A1",
    "market_rent": "1500",
    "current_rent": "1400",
    "tenant": "John Smith",
    "charges": [{"type": "Monthly", "amount": "1400"}, ...],
    "move_in": "2023‑01‑01"
  },
  ...
]
Do NOT summarise or skip units. Output **only JSON**.
PAGE:
--------
{text}
--------
"""

EXTRACTOR_PROMPT_EXCEL = """
You are an expert rent‑roll extractor for Excel‑style tabular chunks. The
Excel text below is tab‑delimited. Extract EVERY unit row and its charges,
returning the **exact** JSON array format described. Do NOT add keys before or
after the array.
TABLE:
--------
{text}
--------
"""

EVALUATOR_PROMPT = """
You validate the Extractor's JSON for completeness and correctness.
Observation text (ground truth):
-----------
{page_text}
-----------
Extractor output:
-----------
{extracted}
-----------
Tasks:
1. Reject if it says e.g. "...and the rest" or omits units/charges.
2. Confirm JSON syntax is an array.
3. (Assume hidden schema) Validate against required keys.
Respond exactly with:
  {"status": "valid"}
OR
  {"status": "invalid", "instructions": "<fix>"}
"""

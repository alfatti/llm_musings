"""
genai_rentroll_agent.py
------------------------------------
End-to-end LangGraph workflow for rent-roll PDF extraction
with Gemini + self-correcting evaluator and JSON-Schema validation.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
from pypdf import PdfReader
import pdfplumber
from jsonschema import validate as json_validate, ValidationError

# ---------- LangChain / LangGraph ----------
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph
from langgraph.graph import StateGraph, END   #  ← NOTE: use the constant END

# -------------------------------------------------------------------------
# 0.  LLM clients
# -------------------------------------------------------------------------
# ──> assumes GOOGLE_API_KEY in env
gemini = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

# -------------------------------------------------------------------------
# 1. Prompt templates
# -------------------------------------------------------------------------
EXTRACTOR_PROMPT = """
You are an expert rent-roll extractor.

Return **ALL** units and charges found in the page below, strictly in JSON
array format (no keys before/after, no comments). Each unit object must
contain these keys:

- "unit"               : string
- "market_rent"        : number or string
- "current_rent"       : number or string
- "tenant"             : string
- "charges"            : array of {{"type": "...", "amount": "..."}}
- "move_in"            : string (YYYY-MM-DD or similar)

PAGE:
--------
{text}
--------
Only output: `[{{...}}, {{...}}]`
"""

EVALUATOR_PROMPT = """
You check the Extractor's JSON for completeness and correctness.

Observation:
-----------
{page_text}
-----------

Extractor output:
-----------
{extracted}
-----------

Tasks:
1. Does the output list **every unit**? Reject if it says e.g. “...and the rest”.
2. For each listed unit, are **all charges** explicitly enumerated?
3. Is it **valid JSON array syntax**?
4. Does it satisfy the JSON schema provided? (do NOT include schema in answer)

If ALL pass, respond with exactly:
    {{"status":"valid"}}

Otherwise respond with:
    {{
       "status":"invalid",
       "instructions":"<<very specific feedback for Extractor to fix>>"
    }}
"""

# -------------------------------------------------------------------------
# 2. JSON-Schema (used both programmatically & referenced implicitly by evaluator)
# -------------------------------------------------------------------------
UNIT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "unit": {"type": "string"},
        "market_rent": {"type": ["string", "number"]},
        "current_rent": {"type": ["string", "number"]},
        "tenant": {"type": "string"},
        "charges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "amount": {"type": ["string", "number"]},
                },
                "required": ["type", "amount"],
            },
        },
        "move_in": {"type": "string"},
    },
    "required": ["unit", "market_rent", "current_rent", "tenant", "charges", "move_in"],
}
PAGE_SCHEMA: Dict[str, Any] = {"type": "array", "items": UNIT_SCHEMA}

# -------------------------------------------------------------------------
# 3.  LLM helper functions
# -------------------------------------------------------------------------
def invoke_gemini(prompt: str) -> str:
    """A thin wrapper so we can swap/chat easily."""
    return gemini.invoke(prompt).content


def extract_units(page_text: str) -> str:
    prompt = EXTRACTOR_PROMPT.format(text=page_text)
    return invoke_gemini(prompt)


def evaluate_output(page_text: str, extracted: str) -> Dict[str, str]:
    """Returns dict with status + optional instructions."""
    prompt = EVALUATOR_PROMPT.format(page_text=page_text, extracted=extracted)
    raw = invoke_gemini(prompt)

    # Must be JSON itself; fall back to invalid if not
    try:
        parsed_eval = json.loads(raw)
        return parsed_eval
    except Exception:
        return {"status": "invalid",
                "instructions": "Evaluator response was not valid JSON."}

# -------------------------------------------------------------------------
# 4. LangGraph state nodes
# -------------------------------------------------------------------------
def extractor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Runs extractor LLM with optional correction instructions."""
    page_text = state["page_text"]
    instructions = state.get("instructions", "")
    if instructions:
        # prepend corrective instructions to steer the LLM
        page_text = f"{instructions}\n\n{page_text}"
    extracted = extract_units(page_text)
    return {**state, "extracted": extracted}

# ───────────────── evaluator_node (robust) ──────────────────
def evaluator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    • Validates the extractor’s raw JSON with three gates:
        1. syntactic JSON.load
        2. JSON-schema validation
        3. semantic/coverage check via LLM evaluator
    • If ANY check fails ➜ status = "invalid", an 'instructions'
      string is added, and the retry counter is incremented.
    """
    page_text: str = state["page_text"]
    extracted_raw: str = state["extracted"]

    # ---------- default result ----------
    eval_result: Dict[str, str] = {"status": "invalid",
                                   "instructions": ""}

    # ---------- 1) JSON syntax ----------
    try:
        extracted_json = json.loads(extracted_raw)
    except Exception as e:
        eval_result["instructions"] = (
            f"Output is not valid JSON. Error: {e}. "
            "Return a proper JSON array like [{{}}, {{}}]."
        )
    else:
        # ---------- 2) JSON-Schema ----------
        try:
            json_validate(instance=extracted_json, schema=PAGE_SCHEMA)
        except ValidationError as ve:
            eval_result["instructions"] = (
                "JSON does not match required schema: "
                f"{ve.message}"
            )
        else:
            # ---------- 3) LLM semantic check ----------
            eval_result = evaluate_output(page_text, extracted_raw)

    # ---------- update state & retries ----------
    new_state = {**state, "eval": eval_result}

    if eval_result.get("status") == "invalid":
        new_state["instructions"] = eval_result.get("instructions", "")
        new_state["retries"] = state.get("retries", 0) + 1

    return new_state
# ─────────────────────────────────────────────────────────────

MAX_RETRIES = 3

def router(state):
    retries = state.get("retries", 0)
    if state["eval"]["status"] == "valid":
        return END                         # ① finish successfully
    elif retries >= MAX_RETRIES:
        print("⚠️ Max retries reached.")
        return END                         # ② finish after too many tries
    else:
        return "extractor"                 # ③ loop back



# -------------------------------------------------------------------------
# 5. Build LangGraph workflow
# -------------------------------------------------------------------------
from typing import TypedDict, Optional, Dict

# Step 1: TypedDict for state
class RentRollState(TypedDict, total=False):
    page_text: str
    extracted: str
    eval: Dict[str, str]
    instructions: Optional[str]
    retries: int

# Step 2: Build graph using schema
graph = StateGraph(state_schema=RentRollState)

graph.add_node("extractor", extractor_node)
graph.add_node("evaluator", evaluator_node)

graph.set_entry_point("extractor")
graph.add_edge("extractor", "evaluator")
graph.add_conditional_edges("evaluator", router)
graph.set_finish_point(END)

workflow = graph.compile()


# -------------------------------------------------------------------------
# 6. PDF helper & run
# -------------------------------------------------------------------------
# def pdf_to_pages(path: str) -> List[str]:
#     reader = PdfReader(path)
#     return [page.extract_text() for page in reader.pages]

def pdf_to_pages(path: str) -> list[str]:
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(layout=True)  # Preserves spacing
            pages.append(text)
    return pages


def process_pdf(path: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for i, page_text in enumerate(pdf_to_pages(path), start=1):
        state = {
            "page_text": page_text,   # mandatory input for extractor
            "retries": 0,             # NEW: retry counter
            # "instructions": ""      # optional; only filled by evaluator
        }
        final = workflow.invoke(state)
        extracted_json = json.loads(final["extracted"])
        # attach page number for traceability
        for unit in extracted_json:
            unit["source_page"] = i
            rows.append(unit)
    return pd.DataFrame(rows)


# -------------------------------------------------------------------------
# 7. Command-line entry point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gemini rent-roll extractor")
    parser.add_argument("pdf_path", help="Path to rent-roll PDF")
    parser.add_argument("-o", "--out", default="rentroll.parquet",
                        help="Output parquet / csv path (inferred by extension)")
    args = parser.parse_args()

    df = process_pdf(args.pdf_path)
    print(df.head())   # quick sanity

    if args.out.endswith(".csv"):
        df.to_csv(args.out, index=False)
    else:
        df.to_parquet(args.out, index=False)

    print(f"✅ Saved {len(df)} rows to {args.out}")

# --------------------------------------------
# Cell 1: Install Dependencies
# --------------------------------------------
!pip install -q langchain langgraph google-generativeai pdfplumber pypdf jsonschema

# --------------------------------------------
# Cell 2: Imports and Setup
# --------------------------------------------
import json
import re
from typing import TypedDict, Optional, List, Union, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from jsonschema import validate as json_validate, ValidationError

# LangChain client setup
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

# --------------------------------------------
# Cell 3: Prompts and Schema
# --------------------------------------------

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

UNIT_SCHEMA = {
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
                "required": ["type", "amount"]
            }
        },
        "move_in": {"type": "string"}
    },
    "required": ["unit", "market_rent", "current_rent", "tenant", "charges", "move_in"]
}
PAGE_SCHEMA = {"type": "array", "items": UNIT_SCHEMA}

# --------------------------------------------
# Cell 4 : LangGraph state + node definitions
# --------------------------------------------
from typing import TypedDict, List, Union, Optional

class AgentState(TypedDict):
    page_text: str
    history: List[Union[HumanMessage, AIMessage]]
    result: Optional[str]
    is_valid: bool
    attempts: int
    errors: List[str]          # ⬅️ verbose failure log

MAX_TRIES = 3                 # central place to change retry cap


# ── Extractor -----------------------------------------------------------
def extractor_node(state: AgentState) -> AgentState:
    text = state["page_text"]

    # if evaluator appended “Please fix:” we reuse that message verbatim,
    # otherwise start with the normal extractor prompt:
    fix_msg = next(
        (m for m in reversed(state["history"]) if isinstance(m, HumanMessage) and m.content.startswith("Please fix:")),
        None,
    )
    prompt = fix_msg.content if fix_msg else EXTRACTOR_PROMPT.format(text=text)

    messages = state["history"] + [HumanMessage(content=prompt)]
    response = llm.invoke(messages)

    return {
        **state,
        "history": messages + [response],
        "result": response.content,
        "attempts": state["attempts"] + 1,
    }


# ── Evaluator -----------------------------------------------------------
def evaluator_node(state: AgentState) -> AgentState:
    text   = state["page_text"]
    output = state["result"]
    errors = []                       # collect granular reasons

    # 1. quick truncation check
    if "and the rest" in output.lower():
        errors.append("Output truncates units with a phrase like 'and the rest…'.")

    # 2. JSON parsing
    parsed = None
    try:
        parsed = json.loads(output)
    except Exception as e:
        errors.append(f"Not valid JSON – {e}")

    # 3. schema validation
    if parsed is not None:
        try:
            json_validate(instance=parsed, schema=PAGE_SCHEMA)
        except ValidationError as e:
            errors.append(f"Schema violation – {e.message}")

    # 4. cross-check *approx* unit count
    if parsed is not None:
        unit_lines = [ln for ln in text.splitlines() if re.match(r"^\s*\w{1,10}\s", ln)]
        if len(parsed) < len(unit_lines):
            errors.append(
                f"Only {len(parsed)} units returned, but ~{len(unit_lines)} detected on page."
            )

    # ---- build state & feedback
    if errors:            # INVALID
        feedback = "Please fix:\n" + "\n".join(f"- {e}" for e in errors)
        state["history"].append(HumanMessage(content=feedback))
        return {**state, "is_valid": False, "errors": state["errors"] + [f"Attempt {state['attempts']}: " + '; '.join(errors)]}

    # VALID
    return {**state, "is_valid": True, "errors": state["errors"]}

# --------------------------------------------
# Cell 5: Build LangGraph
# --------------------------------------------
workflow = StateGraph(AgentState)
workflow.add_node("extract", extractor_node)
workflow.add_node("evaluate", evaluator_node)

workflow.set_entry_point("extract")
workflow.add_edge("extract", "evaluate")
workflow.add_conditional_edges(
    "evaluate",
    lambda s: END if s["is_valid"] or s["attempts"] >= 3 else "extract",
)

graph = workflow.compile()

# ==========================================================
# Cell 8 : Run agent on a single page, record detail
# ==========================================================
def run_agent_on_page(page_text: str) -> dict:
    """
    Returns a dict with:
      • 'valid'     True/False
      • 'attempts'  int
      • 'raw_json'  str  (Gemini’s final answer, valid or not)
      • 'parsed'    list | None
      • 'errors'    list[str]   verbose log per iteration
    """
    init = {
        "page_text": page_text,
        "history": [],
        "result": None,
        "is_valid": False,
        "attempts": 0,
        "errors": [],
    }
    final = graph.invoke(init)

    # After MAX_TRIES, evaluator may still mark invalid; we keep raw result anyway
    raw = final["result"]
    parsed = None
    try:
        parsed = json.loads(raw) if final["is_valid"] else None
    except Exception:
        pass

    return {
        "valid": final["is_valid"],
        "attempts": final["attempts"],
        "raw_json": raw,
        "parsed": parsed,
        "errors": final["errors"],
    }

# ==========================================================
# Cell 9 : Process entire PDF and collect BOTH raw + DataFrame
# ==========================================================
def extract_rentroll_pdf(pdf_path: str | Path):
    """
    Returns (json_pages, dataframe)
      • json_pages : list[str]  – raw extractor output for each page
      • dataframe  : pd.DataFrame of all successfully-parsed units
    Also prints per-page status with verbose evaluator logs.
    """
    json_pages: list[str] = []
    records:    list[dict] = []

    with pdfplumber.open(pdf_path) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            print(f"\n=== Page {idx} ===")
            text = page.extract_text() or ""
            res  = run_agent_on_page(text)

            # keep raw JSON for user inspection
            json_pages.append(res["raw_json"])

            if res["valid"]:
                print(f"✅ Parsed after {res['attempts']} attempt(s). "
                      f"Units extracted: {len(res['parsed'])}")
                for unit in res["parsed"]:
                    unit["_page"] = idx
                    records.append(unit)
            else:
                print(f"❌ Still invalid after {MAX_TRIES} attempts.")
                for err in res["errors"]:
                    print("  •", err)

    df = pd.DataFrame(records)
    return json_pages, df
# ==========================================================
# Cell 10 : Run end-to-end
# ==========================================================
pdf_file = "my_rentroll.pdf"          # 🔄 put your file here

raw_json_list, rentroll_df = extract_rentroll_pdf(pdf_file)

# ---- show DataFrame --------------------------------------
from IPython.display import display, JSON
display(rentroll_df.head())
print(f"\nRows in DataFrame: {len(rentroll_df)}")

# ---- keep raw JSONs handy --------------------------------
print("\n🔎 raw_json_list[0]  (first page excerpt):")
JSON(json.loads(raw_json_list[0][:1000]))              # pretty print first 1 kB



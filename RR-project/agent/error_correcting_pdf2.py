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
# Cell 4: LangGraph State and Node Definitions
# --------------------------------------------

class AgentState(TypedDict):
    page_text: str
    history: List[Union[HumanMessage, AIMessage]]
    result: Optional[str]
    is_valid: bool
    attempts: int

def extractor_node(state: AgentState) -> AgentState:
    text = state["page_text"]
    instructions = state["history"][-1].content if state["history"] else EXTRACTOR_PROMPT.format(text=text)
    prompt = instructions if "Please fix:" in instructions else EXTRACTOR_PROMPT.format(text=text)
    messages = state["history"] + [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    return {
        **state,
        "history": messages + [response],
        "result": response.content,
        "attempts": state["attempts"] + 1,
    }

def evaluator_node(state: AgentState) -> AgentState:
    text = state["page_text"]
    result = state["result"]

    # Quick string check
    if "and the rest" in result.lower():
        state["history"].append(HumanMessage(content="Please fix: output was truncated. List all units explicitly."))
        return {**state, "is_valid": False}

    try:
        parsed = json.loads(result)
    except Exception as e:
        state["history"].append(HumanMessage(content=f"Please fix: Output is not valid JSON. Error: {e}"))
        return {**state, "is_valid": False}

    # JSON Schema Validation
    try:
        json_validate(instance=parsed, schema=PAGE_SCHEMA)
    except ValidationError as e:
        state["history"].append(HumanMessage(content=f"Please fix: JSON does not match required schema. Error: {e.message}"))
        return {**state, "is_valid": False}

    return {**state, "is_valid": True}

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

# --------------------------------------------
# Cell 6: Try on a Sample Page
# --------------------------------------------
sample_page = """
Unit Mkt Rent Curr Rent Lease Primary Tenant Next Date Description Monthly GPR
A101 $2000 $1950 John Doe 6/1/2024 1BD/1BA 1950 1950
A102 $2100 $2050 Jane Smith 6/1/2024 2BD/1BA 2050 2050
A103 $2200 $2150 Mike Ross 6/1/2024 2BD/2BA 2150 2150
"""

initial_state = {
    "page_text": sample_page,
    "history": [],
    "result": None,
    "is_valid": False,
    "attempts": 0,
}

final_state = graph.invoke(initial_state)

# --------------------------------------------
# Cell 7: Final Output
# --------------------------------------------
print("✅ Final JSON output:")
print(final_state["result"])
print("\nAttempts:", final_state["attempts"])

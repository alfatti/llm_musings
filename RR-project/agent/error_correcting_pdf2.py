# Step 1: Imports and setup
!pip install -q langgraph langchain google-generativeai

import json
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import END, StateGraph
from typing import TypedDict, Annotated, List, Optional, Union
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import Runnable
import re

# Step 2: LLM Setup (Extractor)
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

# Step 3: Define the agent state
class AgentState(TypedDict):
    page_text: str
    history: List[Union[HumanMessage, AIMessage]]
    result: Optional[str]
    is_valid: bool
    attempts: int

# Step 4: Define Extractor Node
def extractor_node(state: AgentState) -> AgentState:
    prompt = (
        "You are a rent roll data extractor. "
        "You must extract all units and charges shown in the text. "
        "Each unit should be a dictionary with its information, "
        "and all dictionaries should be returned as a list in strict JSON format: [{}, {}, ...]. "
        "Do NOT summarize or skip units. Output only the JSON.\n\n"
        f"Page content:\n{state['page_text']}"
    )

    messages = state['history'] + [HumanMessage(content=prompt)]
    response = llm.invoke(messages)

    return {
        **state,
        "history": messages + [response],
        "result": response.content,
        "attempts": state["attempts"] + 1,
    }

# Step 5: Define Evaluator Node
def evaluator_node(state: AgentState) -> AgentState:
    result = state["result"]
    text = state["page_text"]
    is_valid = True
    feedback = ""

    # JSON check
    try:
        parsed = json.loads(result)
        if not isinstance(parsed, list) or not all(isinstance(x, dict) for x in parsed):
            raise ValueError("Not a list of dicts.")
    except Exception as e:
        feedback += f"Output is not valid JSON. Error: {e}\n"
        is_valid = False

    # Count unit names in page text
    unit_lines = [line for line in text.splitlines() if re.match(r"^\s*\w{1,10}\s", line)]
    expected_units = len(unit_lines)
    actual_units = len(parsed) if is_valid else 0

    if actual_units < expected_units:
        feedback += (
            f"Expected ~{expected_units} units based on the page text, but only {actual_units} were found.\n"
            "Do not truncate the output or say 'and the rest are similar'. List all units explicitly.\n"
        )
        is_valid = False

    # If invalid, append feedback as a HumanMessage
    if not is_valid:
        state["history"].append(HumanMessage(content=f"Please fix: {feedback}"))
        return {**state, "is_valid": False}
    
    return {**state, "is_valid": True}

# Step 6: Build LangGraph
workflow = StateGraph(AgentState)

workflow.add_node("extract", extractor_node)
workflow.add_node("evaluate", evaluator_node)

# Edges
workflow.set_entry_point("extract")
workflow.add_edge("extract", "evaluate")
workflow.add_conditional_edges(
    "evaluate",
    lambda state: END if state["is_valid"] or state["attempts"] >= 3 else "extract",
)

graph = workflow.compile()

# Step 7: Run on a sample page
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

# Step 8: Display Result
from pprint import pprint
print("✅ Final JSON output:")
pprint(final_state["result"])

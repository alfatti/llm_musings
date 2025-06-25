### file: graph_builder.py
from langgraph.graph import StateGraph

from extractor import Extractor
from evaluator import Evaluator

_ext = Extractor()
_eval = Evaluator()


def extractor_node(state):
    text = state["chunk_text"]
    file_type = state["file_type"]
    instructions = state.get("instructions", "")
    if instructions:
        text = f"{instructions}\n\n{text}"
    extracted = _ext.extract(text, file_type)
    return {**state, "extracted": extracted}


def evaluator_node(state):
    result = _eval.evaluate(state["chunk_text"], state["extracted"])
    return {**state, "eval": result}


def router(state):
    return "end" if state["eval"]["status"] == "valid" else "extractor"


def build_workflow():
    g = StateGraph()
    g.add_node("extractor", extractor_node)
    g.add_node("evaluator", evaluator_node)
    g.set_entry_point("extractor")
    g.add_edge("extractor", "evaluator")
    g.add_conditional_edges("evaluator", router)
    g.set_finish_point("end")
    return g.compile()

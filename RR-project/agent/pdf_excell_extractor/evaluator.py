### file: evaluator.py
import json
from jsonschema import validate, ValidationError
from langchain_google_genai import ChatGoogleGenerativeAI
from prompts import EVALUATOR_PROMPT

# ------------- JSON Schema -------------
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
                "required": ["type", "amount"],
            },
        },
        "move_in": {"type": "string"},
    },
    "required": [
        "unit",
        "market_rent",
        "current_rent",
        "tenant",
        "charges",
        "move_in",
    ],
}
PAGE_SCHEMA = {"type": "array", "items": UNIT_SCHEMA}


class Evaluator:
    """Schema + semantic validator with corrective feedback."""

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.0)

    # --- helper ----
    def _llm_semantic_check(self, page_text: str, extracted: str) -> dict:
        prompt = EVALUATOR_PROMPT.format(page_text=page_text, extracted=extracted)
        raw = self.llm.invoke(prompt).content
        try:
            return json.loads(raw)
        except Exception:
            return {
                "status": "invalid",
                "instructions": "Evaluator did not return valid JSON.",
            }

    # --- public ----
    def evaluate(self, page_text: str, extracted: str) -> dict:
        # 1) JSON syntax
        try:
            data = json.loads(extracted)
        except Exception as e:
            return {"status": "invalid", "instructions": f"JSON parse error: {e}"}

        # 2) Schema
        try:
            validate(instance=data, schema=PAGE_SCHEMA)
        except ValidationError as ve:
            return {
                "status": "invalid",
                "instructions": f"Schema violation: {ve.message}",
            }

        # 3) Semantic completeness
        return self._llm_semantic_check(page_text, extracted)


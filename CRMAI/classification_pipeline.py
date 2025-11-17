import csv
import json
from dataclasses import dataclass
from typing import List, Dict, Any

from openai import OpenAI

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

# Swap this to whatever model name your environment exposes for Gemini 2.5 Pro
GEMINI_MODEL = "google/gemini-2.5-pro"

# Define the allowed categories for the PoC
ALLOWED_CATEGORIES = [
    "Account Opening",
    "Signature Maintenance",
    "Representative Update",
    "Treasury / ACH Issue",
    "Credit Line Inquiry",
    "KYC / Documentation Update",
    "Access / Portal Issue",
    "Other",
]


# -------------------------------------------------------------------
# Data structures & loading
# -------------------------------------------------------------------

@dataclass
class LabeledEmail:
    email: str
    category: str
    task_notes: str


def load_few_shot_examples(csv_path: str) -> List[LabeledEmail]:
    """
    Load pre-sampled few-shot examples from a small CSV file.

    Expected columns: 'email', 'category', 'task_notes'
    """
    examples: List[LabeledEmail] = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            examples.append(
                LabeledEmail(
                    email=row["email"],
                    category=row["category"],
                    task_notes=row["task_notes"],
                )
            )

    return examples


# -------------------------------------------------------------------
# Prompt builder
# -------------------------------------------------------------------

def build_few_shot_block(examples: List[LabeledEmail]) -> str:
    """
    Turn the list of LabeledEmail examples into a formatted block
    for inclusion in the few-shot prompt.
    """
    blocks = []
    for ex in examples:
        block = f"""EXAMPLE
Client Email:
\"\"\"{ex.email.strip()}\"\"\"

Category: {ex.category.strip()}
Task Notes:
\"\"\"{ex.task_notes.strip()}\"\"\""""
        blocks.append(block)

    return "\n\n".join(blocks)


def build_prompt(new_email: str, examples: List[LabeledEmail]) -> str:
    """
    Build the full few-shot prompt (single string) for one new email.
    """
    few_shot_text = build_few_shot_block(examples)
    categories_text = "\n".join(f"- {c}" for c in ALLOWED_CATEGORIES)

    prompt = f"""
You are an expert Relationship Associate in Corporate & Investment Banking.
Your job is to:
1) Assign the correct case category for an incoming client email.
2) Draft clear, concise, professional task notes for the Middle Office.
3) Estimate a numeric confidence between 0 and 1.

Use ONLY the following allowed categories (pick the best single one):
{categories_text}

Output STRICTLY as a compact JSON object with keys:
- "category": one of the allowed categories
- "confidence": a number between 0 and 1 (float)
- "task_notes": a short paragraph of instructions for the Middle Office

Guidelines for task_notes:
- Write in third person, action-oriented language.
- Include all key details from the email (accounts, entities, requested change).
- Be concise but specific enough for the Middle Office to action without re-reading the original email.

Use the labeled examples below as guidance:

{few_shot_text}

Now classify the NEW email below and produce ONLY the JSON object
(with no extra commentary or explanations).

NEW EMAIL:
\"\"\"{new_email.strip()}\"\"\""""

    return prompt.strip()


# -------------------------------------------------------------------
# LLM call + parsing pipeline
# -------------------------------------------------------------------

def parse_json_from_response(response) -> Dict[str, Any]:
    """
    Extract and parse the JSON string from a Responses API response.
    Handles both raw string and .value cases.
    """
    # For the Responses API, we expect something like:
    # response.output[0].content[0].text (which may have a .value attribute)
    content = response.output[0].content[0].text

    # Depending on the client version, `text` may be a plain string or an object with `.value`
    if hasattr(content, "value"):
        json_str = content.value
    else:
        json_str = content

    return json.loads(json_str)


def classify_email(
    email_text: str,
    examples_csv_path: str,
    client: OpenAI,
    model: str = GEMINI_MODEL,
) -> Dict[str, Any]:
    """
    End-to-end pipeline:
    - Load few-shot examples from CSV (already pre-sampled for this PoC)
    - Build the prompt
    - Call Gemini via OpenAI client
    - Parse JSON (category, confidence, task_notes)

    Returns a dict:
    {
      "category": str,
      "confidence": float,
      "task_notes": str,
      "raw": <original JSON from model>
    }
    """
    # 1) Load few-shot examples (small CSV, already curated)
    examples = load_few_shot_examples(examples_csv_path)

    # 2) Build few-shot prompt for this specific email
    prompt = build_prompt(email_text, examples)

    # 3) Call the model via Responses API
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt,
                    }
                ],
            }
        ],
        # Ask model to return a JSON object; it will still be text,
        # but constrained to valid JSON.
        response_format={"type": "json_object"},
    )

    # 4) Parse JSON and apply basic post-processing
    json_obj = parse_json_from_response(response)

    category = json_obj.get("category", "Other")
    confidence_raw = json_obj.get("confidence", 0.0)
    task_notes = json_obj.get("task_notes", "").strip()

    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 0.0

    result = {
        "category": category,
        "confidence": confidence,
        "task_notes": task_notes,
        "raw": json_obj,
    }

    return result


# -------------------------------------------------------------------
# Example usage (for your notebook / script)
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Instantiate the OpenAI client (you already have auth/env wired up)
    client = OpenAI()

    # Path to your small sampled CSV of (email, category, task_notes)
    examples_csv = "few_shot_cases_sampled.csv"

    # Example email text (in practice this comes from Outlook / NOW export)
    incoming_email = """
    Hi team,

    We have a new authorized signer for ABC Holdings LLC and need to update the
    signature card and access profiles for our operating accounts ending 1234 and 5678.

    Please remove John Doe and add Jane Smith as the sole authorized signer.

    Thanks,
    Client
    """

    result = classify_email(
        email_text=incoming_email,
        examples_csv_path=examples_csv,
        client=client,
    )

    print("Predicted category:", result["category"])
    print("Confidence:", result["confidence"])
    print("Proposed task notes:\n", result["task_notes"])

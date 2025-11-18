import os
import csv
import json
from dataclasses import dataclass
from typing import List, Dict, Any

import streamlit as st
from openai import OpenAI

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

GEMINI_MODEL = "google/gemini-2.5-pro"  # change if your env uses a different alias

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


@st.cache_data(show_spinner=False)
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
(with no extra commentary, no markdown, no backticks).

NEW EMAIL:
\"\"\"{new_email.strip()}\"\"\""""
    return prompt.strip()


# -------------------------------------------------------------------
# LLM call + parsing
# -------------------------------------------------------------------

def parse_json_from_response(response) -> Dict[str, Any]:
    """
    Extract and parse the JSON string from a Responses API response.
    Handles both raw string and .value cases.
    """
    content = response.output[0].content[0].text
    if hasattr(content, "value"):
        json_str = content.value
    else:
        json_str = content

    # Try to trim any junk before/after JSON (defensive)
    json_str = json_str.strip()
    # Simple guard in case the model adds leading/trailing text
    if json_str.startswith("```"):
        json_str = json_str.strip("`")
    return json.loads(json_str)


def classify_email_with_examples(
    email_text: str,
    examples: List[LabeledEmail],
    client: OpenAI,
    model: str = GEMINI_MODEL,
) -> Dict[str, Any]:
    """
    End-to-end classification given already-loaded examples.
    Returns:
    {
      "category": str,
      "confidence": float,
      "task_notes": str,
      "raw": <original JSON from model>
    }
    """
    prompt = build_prompt(email_text, examples)

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
        temperature=0.1,
        max_output_tokens=512,
    )

    json_obj = parse_json_from_response(response)

    category = json_obj.get("category", "Other")
    task_notes = (json_obj.get("task_notes") or "").strip()

    try:
        confidence = float(json_obj.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    return {
        "category": category,
        "confidence": confidence,
        "task_notes": task_notes,
        "raw": json_obj,
    }


# -------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Client Email Triage – PoC",
        layout="wide",
    )

    st.title("📨 Client Email Triage – PoC")
    st.caption(
        "Paste an incoming client email and get a proposed case category, "
        "confidence score, and task notes for the Middle Office."
    )

    # Sidebar config
    with st.sidebar:
        st.header("⚙️ Configuration")

        default_csv = "few_shot_cases_sampled.csv"
        csv_path = st.text_input(
            "Few-shot examples CSV path",
            value=default_csv,
            help="Small CSV with columns: email, category, task_notes",
        )

        model_name = st.text_input(
            "Model name",
            value=GEMINI_MODEL,
            help="Gemini 2.5 Pro alias / model name exposed via OpenAI client.",
        )

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05,
        )

        st.markdown("---")
        st.markdown(
            "This is a **PoC-only** tool, not connected to production systems."
        )

    # Main input area
    st.subheader("✉️ Paste Email Text")
    email_text = st.text_area(
        "Client Email Body",
        height=260,
        placeholder=(
            "Hi team,\n\nWe have a new authorized signer for ABC Holdings LLC and "
            "need to update the signature card and access profiles..."
        ),
    )

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        classify_clicked = st.button("🔍 Classify Email", type="primary")
    with col_info:
        st.write(
            "The model will infer a **case category**, approximate **confidence**, "
            "and **task notes** for Middle Office fulfillment."
        )

    if classify_clicked:
        if not email_text.strip():
            st.warning("Please paste an email body first.")
            return

        if not os.path.exists(csv_path):
            st.error(f"Few-shot CSV not found at: `{csv_path}`")
            return

        try:
            with st.spinner("Classifying email..."):
                examples = load_few_shot_examples(csv_path)

                # Instantiate client (assumes env vars already set)
                client = OpenAI()

                # Use the selected model and temperature
                result = classify_email_with_examples(
                    email_text=email_text,
                    examples=examples,
                    client=client,
                    model=model_name,
                )

            # Display results
            st.markdown("## ✅ Model Output")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Proposed Category**")
                st.markdown(
                    f"<div style='padding:0.5rem 0.75rem; "
                    f"border-radius:0.5rem; background-color:#f0f2f6; "
                    f"display:inline-block; font-weight:600;'>"
                    f"{result['category']}</div>",
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown("**Confidence**")
                conf_pct = max(0.0, min(1.0, result["confidence"])) * 100
                st.metric(label="Confidence (0–100%)", value=f"{conf_pct:.1f} %")

            st.markdown("### 📝 Proposed Task Notes")
            st.write(result["task_notes"] or "_(Task notes were empty)_")

            with st.expander("🔎 Raw JSON (debug)"):
                st.json(result["raw"])

        except Exception as e:
            st.error(f"Error while classifying email: {e}")


if __name__ == "__main__":
    main()

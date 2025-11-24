#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Case email classifier using Gemini via Apigee/OpenAI-style client.

Pipeline:
- Read a small CSV of labeled examples (email_text, category, task_notes)
- Build a few-shot prompt
- For a new incoming email, call Gemini to:
    * predict category
    * propose task_notes
    * output a confidence score in [0, 1]
- Uses OpenAI client with Apigee base_url + access token
- Uses the same call_gemini_openai_style + assemble_headers pattern as your working script

Example usage:

  python case_email_classifier.py \
      --examples-csv examples.csv \
      --email-text "Hi, we need to amend the notional on the swap..." \
      --email-col email_text \
      --category-col category \
      --notes-col task_notes \
      --max-examples 8 \
      --model gemini-2.5.pro \
      --apigee-base-url "https://<company>.../v1/openai/v1"
"""

import argparse
import json
import os
import re
import uuid
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ---------- Utilities ----------

def generate_uuid() -> str:
    return str(uuid.uuid4())

def iso_now_utc() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

# ---------- Header assembly (no Authorization) ----------

def assemble_headers(
    api_key: str,
    usecase_id: str,
    app_id: str,
) -> Dict[str, str]:
    base_headers = {
        "x-wf-api-key": api_key,
        "x-wf-usecase-id": usecase_id,
        "x-wf-client-id": app_id,
    }
    request_id = correlation_id = generate_uuid()
    request_date = iso_now_utc()

    merged = {
        **base_headers,
        "x-request-id": request_id,
        "x-wf-request-date": request_date,
        "x-correlation-id": correlation_id,
        "Content-Type": "application/json",
    }
    return merged

# ---------- LLM call (mimicking your working pattern) ----------

def call_gemini_openai_style(
    openai_client: Any,
    model: str,
    system_text: str,
    user_payload: Dict[str, Any],
    headers: Dict[str, str],
    temperature: float = 0.0,
    max_tokens: int = 800,
) -> Tuple[Optional[str], Any]:
    """
    Text-only variant of your working call_gemini_openai_style.

    - Uses the same structure: system message + user message with JSON-serialized payload
    - Passes semantic headers via extra_headers
    - No Authorization header here; it's handled by OpenAI(api_key=apigee_access_token)
    """
    messages = [
        {"role": "system", "content": system_text},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(user_payload, ensure_ascii=False),
                }
            ],
        },
    ]

    resp = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_headers=headers,
    )
    content = resp.choices[0].message.content if (resp and getattr(resp, "choices", None)) else None
    return content, resp

# ---------- Simple, reasonably robust JSON parser ----------

def coerce_json_from_text(text: Optional[str]) -> Dict[str, Any]:
    """
    Try to extract a JSON object from the model output.
    - Strip code fences
    - Try full string
    - Try first {...} block
    """
    if text is None:
        raise ValueError("Empty completion text.")

    raw = text.strip()
    # remove ```json ... ``` and ``` ... ```
    raw = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).replace("```", "").strip()

    # Direct attempt
    try:
        return json.loads(raw)
    except Exception:
        pass

    # Try to extract first {...}
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        candidate = raw[start:end+1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # last attempt: basic replacements for smart quotes
    cleaned = (
        raw.replace("“", '"')
           .replace("”", '"')
           .replace("’", "'")
           .replace("`", '"')
    )
    try:
        return json.loads(cleaned)
    except Exception as e:
        raise ValueError(f"Could not parse JSON from model output: {e}\nRAW:\n{text}") from e

# ---------- Few-shot prompt builder ----------

def build_few_shot_prompt(
    df: pd.DataFrame,
    email_col: str,
    category_col: str,
    notes_col: str,
    max_examples: int = 10,
) -> str:
    """
    Build a few-shot block from labeled examples.
    Format:

    Example 1
    EMAIL:
    <email text>

    CATEGORY: <category>
    TASK_NOTES: <task_notes>
    ----

    ...
    """
    examples = []
    df = df[[email_col, category_col, notes_col]].dropna().head(max_examples)

    for i, row in df.iterrows():
        email_text = str(row[email_col]).strip()
        category = str(row[category_col]).strip()
        notes = str(row[notes_col]).strip()

        block = (
            f"Example {len(examples) + 1}\n"
            f"EMAIL:\n{email_text}\n\n"
            f"CATEGORY: {category}\n"
            f"TASK_NOTES: {notes}\n"
            f"----"
        )
        examples.append(block)

    return "\n\n".join(examples)

# ---------- Prompt builder for classification ----------

def build_prompt_for_email(
    few_shot_block: str,
    incoming_email: str,
) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (system_text, user_payload) for the classifier call.
    The model is instructed to output STRICT JSON ONLY with:
      { "category": "...", "task_notes": "...", "confidence": 0.0-1.0 }
    """

    system_text = (
        "You are an assistant that reads client servicing emails for a corporate banking middle office. "
        "Your job is to assign a case category and propose concise operational task notes. "
        "Always respond in STRICT JSON only, with fields: category, task_notes, confidence."
    )

    output_schema = {
        "type": "object",
        "required": ["category", "task_notes", "confidence"],
        "properties": {
            "category": {
                "type": "string",
                "description": "Predicted case category label, similar to the training examples."
            },
            "task_notes": {
                "type": "string",
                "description": "Concise instructions to the middle office on what to do."
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Model's confidence in the category (0=low,1=high)."
            },
        },
        "additionalProperties": False,
    }

    user_payload: Dict[str, Any] = {
        "instructions": (
            "You are given a few labeled examples of client emails and their categories/task notes. "
            "Use them as guidance to classify a new email."
        ),
        "few_shot_examples": few_shot_block,
        "incoming_email": incoming_email,
        "output_schema": output_schema,
        "formatting": (
            "Return STRICT JSON ONLY, no markdown, no backticks. "
            "Fields: category (string), task_notes (string), confidence (float in [0,1])."
        ),
        "example_output": {
            "category": "AMENDMENT - SWAP NOTIONAL",
            "task_notes": "Request trade capture to amend notional as per client instruction and send revised confirmation.",
            "confidence": 0.86,
        },
    }

    return system_text, user_payload

# ---------- Orchestration ----------

def classify_email(
    openai_client: Any,
    model: str,
    api_key: str,
    usecase_id: str,
    app_id: str,
    few_shot_block: str,
    incoming_email: str,
) -> Dict[str, Any]:
    system_text, user_payload = build_prompt_for_email(
        few_shot_block=few_shot_block,
        incoming_email=incoming_email,
    )

    headers = assemble_headers(api_key=api_key, usecase_id=usecase_id, app_id=app_id)

    raw_text, raw_resp = call_gemini_openai_style(
        openai_client=openai_client,
        model=model,
        system_text=system_text,
        user_payload=user_payload,
        headers=headers,
        temperature=0.0,
        max_tokens=400,
    )

    parsed = coerce_json_from_text(raw_text)

    # Optional: minimal sanity check
    for key in ("category", "task_notes", "confidence"):
        if key not in parsed:
            raise ValueError(f"Model output missing '{key}': {parsed}")

    return {
        "raw_text": raw_text,
        "parsed": parsed,
        "request_headers_used": headers,
    }

# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(description="Case email classifier via Gemini (OpenAI-style, Apigee).")

    parser.add_argument("--examples-csv", required=True, help="Path to CSV of labeled examples.")
    parser.add_argument("--email-col", default="email_text", help="Column name for email text in the examples CSV.")
    parser.add_argument("--category-col", default="category", help="Column name for category in the examples CSV.")
    parser.add_argument("--notes-col", default="task_notes", help="Column name for task notes in the examples CSV.")
    parser.add_argument("--max-examples", type=int, default=10, help="Max number of few-shot examples to use.")
    parser.add_argument("--email-text", required=True, help="Incoming email text to classify.")
    parser.add_argument("--model", default="gemini-2.5.pro", help="Model name.")

    # creds & routing
    parser.add_argument("--api-key", default=os.getenv("WF_API_KEY"), help="x-wf-api-key")
    parser.add_argument("--usecase-id", default=os.getenv("WF_USECASE_ID"), help="x-wf-usecase-id")
    parser.add_argument("--app-id", default=os.getenv("WF_APP_ID"), help="x-wf-client-id")
    parser.add_argument("--apigee-access-token", default=os.getenv("APIGEE_ACCESS_TOKEN"), help="Apigee access token")
    parser.add_argument("--apigee-base-url", default=os.getenv("APIGEE_BASE_URL"), help="Apigee base URL for OpenAI proxy")

    args = parser.parse_args()

    missing = []
    if not args.api_key: missing.append("WF_API_KEY / --api-key")
    if not args.usecase_id: missing.append("WF_USECASE_ID / --usecase-id")
    if not args.app_id: missing.append("WF_APP_ID / --app-id")
    if not args.apigee_access_token: missing.append("APIGEE_ACCESS_TOKEN / --apigee-access-token")
    if not args.apigee_base_url: missing.append("APIGEE_BASE_URL / --apigee-base-url")
    if missing:
        raise SystemExit("Missing: " + ", ".join(missing))

    # Construct OpenAI client with Apigee proxy routing
    try:
        from openai import OpenAI
        openai_client = OpenAI(
            api_key=args.apigee_access_token,
            base_url=args.apigee_base_url,
        )
    except Exception as e:
        raise SystemExit(f"Failed to construct OpenAI client with Apigee routing: {e}")

    # Load examples and build few-shot prompt
    df = pd.read_csv(args.examples_csv)
    few_shot_block = build_few_shot_prompt(
        df=df,
        email_col=args.email_col,
        category_col=args.category_col,
        notes_col=args.notes_col,
        max_examples=args.max_examples,
    )

    result = classify_email(
        openai_client=openai_client,
        model=args.model,
        api_key=args.api_key,
        usecase_id=args.usecase_id,
        app_id=args.app_id,
        few_shot_block=few_shot_block,
        incoming_email=args.email_text,
    )

    print("\n=== RAW MODEL TEXT ===\n")
    print(result["raw_text"])
    print("\n=== PARSED JSON ===\n")
    print(json.dumps(result["parsed"], indent=2, ensure_ascii=False))
    print("\n=== REQUEST HEADER SUMMARY (no bearer) ===\n")
    print(json.dumps(result["request_headers_used"], indent=2))

if __name__ == "__main__":
    main()

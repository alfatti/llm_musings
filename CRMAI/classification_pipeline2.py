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

import re

def parse_classifier_output_text(raw_text: Optional[str]) -> Dict[str, Any]:
    if not raw_text:
        raise ValueError("Empty completion text from model.")

    # Remove accidental fences
    txt = raw_text.strip()
    if txt.startswith("```"):
        txt = txt.strip("`").strip()

    # Extract lines
    cat_match = re.search(r"^CATEGORY:\s*(.+)$", txt, flags=re.MULTILINE | re.IGNORECASE)
    notes_match = re.search(r"^TASK_NOTES:\s*(.+)$", txt, flags=re.MULTILINE | re.IGNORECASE)
    conf_match = re.search(r"^CONFIDENCE:\s*([0-9]*\.?[0-9]+)", txt, flags=re.MULTILINE | re.IGNORECASE)

    if not cat_match or not notes_match or not conf_match:
        raise ValueError(f"Could not parse classifier output.\nRAW:\n{raw_text}")

    category = cat_match.group(1).strip()
    task_notes = notes_match.group(1).strip()
    confidence = float(conf_match.group(1))

    # Clamp confidence just in case
    confidence = max(0.0, min(1.0, confidence))

    return {
        "category": category,
        "task_notes": task_notes,
        "confidence": confidence,
    }


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

    Output format (STRICT):
      CATEGORY: <category>
      TASK_NOTES: <task_notes on ONE line>
      CONFIDENCE: <float between 0 and 1>

    No JSON, no markdown, no extra commentary.
    """

    system_text = (
        "You are an assistant that reads client servicing emails for a corporate banking middle office. "
        "Your job is to assign a case category and propose concise operational task notes. "
        "Always respond with exactly three lines:\n"
        "CATEGORY: <category>\n"
        'TASK_NOTES: <one-line task notes>\n'
        "CONFIDENCE: <float between 0 and 1>\n"
        "Do not include JSON, markdown, or any extra text."
    )

    user_payload: Dict[str, Any] = {
        "instructions": (
            "You are given a few labeled examples of client emails and their categories/task notes. "
            "Use them as guidance to classify a new email."
        ),
        "few_shot_examples": few_shot_block,
        "incoming_email": incoming_email,
        "formatting": (
            "Return EXACTLY three lines:\n"
            "CATEGORY: <category>\n"
            "TASK_NOTES: <one-line task notes>\n"
            "CONFIDENCE: <float between 0 and 1>\n"
            "No other text, no JSON, no markdown, no bullet points."
        ),
        "example_output": (
            "CATEGORY: AMENDMENT - SWAP NOTIONAL\n"
            "TASK_NOTES: Request trade capture to amend notional as per client instruction and send revised confirmation.\n"
            "CONFIDENCE: 0.86"
        ),
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

    parsed = parse_classifier_output_text(raw_text)

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

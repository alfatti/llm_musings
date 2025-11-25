#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Case email classifier using Gemini via Apigee/OpenAI-style client.

Pipeline:
- Read a small CSV of labeled examples (email_text, category, task_notes)
- Build a few-shot text prompt
- For a new incoming email, call Gemini to:
    * predict category (constrained to an allowed list)
    * propose task_notes
    * output a confidence score in [0, 1]
- Uses OpenAI client with Apigee base_url + access token
- Output format from LLM: 3 plain-text lines (CATEGORY / TASK_NOTES / CONFIDENCE)
- Final category is snapped to an ALLOWED_CATEGORIES list you provide.
"""

import argparse
import json
import os
import re
import uuid
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from difflib import SequenceMatcher


# ---------- Utilities ----------

def generate_uuid() -> str:
    return str(uuid.uuid4())

def iso_now_utc() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


# ---------- Header assembly (no Authorization here) ----------

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
    user_text: str,
    headers: Dict[str, str],
    temperature: float = 0.0,
    max_tokens: int = 400,
) -> Tuple[Optional[str], Any]:
    """
    Text-only call using the same structure as your working snippet:

    - System message (format + role)
    - User message with a single text chunk (few-shot + instructions + new email)
    - Apigee semantic headers via extra_headers
    - No response_format, no Authorization header here (SDK uses api_key)
    """
    messages = [
        {"role": "system", "content": system_text},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_text,
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


# ---------- Parser for 3-line text format (with JSON fallback) ----------

def parse_classifier_output(raw_text: Optional[str]) -> Dict[str, Any]:
    """
    Hybrid parser:
      1. If the output looks like JSON and has category/task_notes/confidence, parse as JSON.
      2. Otherwise, parse plain text with CATEGORY/TASK_NOTES/CONFIDENCE tags.
    """
    if not raw_text:
        raise ValueError("Empty completion text from model.")

    txt = raw_text.strip()

    # Strip accidental fences
    if txt.startswith("```"):
        txt = txt.strip("`").strip()

    # --- 1) JSON path if it looks like JSON ---
    looks_like_json = txt.lstrip().startswith("{") or '"category"' in txt.lower()
    if looks_like_json:
        try:
            data = json.loads(txt)
            if all(k in data for k in ("category", "task_notes", "confidence")):
                try:
                    conf = float(data["confidence"])
                except Exception:
                    conf = 0.0
                conf = max(0.0, min(1.0, conf))
                return {
                    "category": str(data["category"]).strip(),
                    "task_notes": str(data["task_notes"]).strip(),
                    "confidence": conf,
                }
        except Exception:
            # fall through to text parsing
            pass

    # --- 2) Plain-text tagged parsing ---
    cat_match = re.search(r"CATEGORY\s*:\s*(.+)", txt, flags=re.IGNORECASE)
    notes_match = re.search(r"TASK[_\s]*NOTES\s*:\s*(.+)", txt, flags=re.IGNORECASE)
    conf_match = re.search(r"CONFIDENCE\s*:\s*([0-9]*\.?[0-9]+)", txt, flags=re.IGNORECASE)

    if not cat_match or not notes_match or not conf_match:
        raise ValueError("Could not parse classifier output.")

    category = cat_match.group(1).strip()
    task_notes = notes_match.group(1).strip()
    confidence = float(conf_match.group(1))
    confidence = max(0.0, min(1.0, confidence))

    return {
        "category": category,
        "task_notes": task_notes,
        "confidence": confidence,
    }


# ---------- Category snapping to allowed list ----------

def normalize_category(
    raw_category: str,
    allowed_categories: List[str],
) -> Tuple[str, float, bool]:
    """
    Snap the raw model category to the closest allowed category.

    Returns:
        (normalized_category, similarity_score, is_exact_match)
    """
    if not allowed_categories:
        # no constraints
        return raw_category.strip(), 0.0, False

    raw = raw_category.strip()
    raw_lower = raw.lower()

    # First try exact (case-insensitive) match
    for cat in allowed_categories:
        if raw_lower == cat.lower().strip():
            return cat.strip(), 1.0, True

    # Otherwise, pick max similarity
    best_cat = allowed_categories[0]
    best_score = 0.0
    for cat in allowed_categories:
        s = SequenceMatcher(None, raw_lower, cat.lower().strip()).ratio()
        if s > best_score:
            best_score = s
            best_cat = cat

    return best_cat.strip(), best_score, False


# ---------- Few-shot prompt builder ----------

def build_few_shot_block(
    df: pd.DataFrame,
    email_col: str,
    category_col: str,
    notes_col: str,
    max_examples: int = 5,
) -> str:
    """
    Build a compact few-shot block from labeled examples.

    Format:

    Example 1
    EMAIL:
    <email text>

    CATEGORY: <category>
    TASK_NOTES: <task notes>
    ----
    """
    examples: List[str] = []
    df = df[[email_col, category_col, notes_col]].dropna().head(max_examples)

    for _, row in df.iterrows():
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
    allowed_categories: List[str],
) -> Tuple[str, str]:
    """
    Returns (system_text, user_text) for the classifier call.

    Output format (STRICT):
      CATEGORY: <category from allowed list>
      TASK_NOTES: <task_notes on ONE line>
      CONFIDENCE: <float between 0 and 1>
    """

    allowed_block = "\n".join(f"- {c}" for c in allowed_categories)

    system_text = (
        "You are an assistant that reads client servicing emails for a corporate banking middle office. "
        "Your job is to assign a case category and propose concise operational task notes. "
        "You MUST choose the CATEGORY from the allowed list provided. "
        "You MUST respond using exactly three lines:\n"
        "CATEGORY: <category from allowed list>\n"
        "TASK_NOTES: <one-line task notes>\n"
        "CONFIDENCE: <float between 0 and 1>\n"
        "Do not include JSON, markdown, bullet points, or any extra commentary."
    )

    user_text = (
        "You will see some labeled examples of client emails, each with CATEGORY and TASK_NOTES.\n\n"
        "Use them as guidance to classify a NEW email.\n\n"
        "=== ALLOWED CATEGORIES ===\n"
        f"{allowed_block}\n\n"
        "You MUST choose CATEGORY from this list exactly. If the email is ambiguous, pick the closest matching category.\n\n"
        "=== LABELED EXAMPLES ===\n"
        f"{few_shot_block}\n\n"
        "=== NEW EMAIL TO CLASSIFY ===\n"
        f"EMAIL:\n{incoming_email}\n\n"
        "=== REQUIRED OUTPUT FORMAT ===\n"
        "Respond with EXACTLY three lines, no extra text:\n"
        "CATEGORY: <one of the allowed categories>\n"
        "TASK_NOTES: <one-line task notes>\n"
        "CONFIDENCE: <float between 0 and 1>\n"
        "Do not omit TASK_NOTES or CONFIDENCE. Do not add any additional lines."
    )

    return system_text, user_text


# ---------- Orchestration ----------

def classify_email(
    openai_client: Any,
    model: str,
    api_key: str,
    usecase_id: str,
    app_id: str,
    few_shot_block: str,
    incoming_email: str,
    allowed_categories: List[str],
) -> Dict[str, Any]:
    system_text, user_text = build_prompt_for_email(
        few_shot_block=few_shot_block,
        incoming_email=incoming_email,
        allowed_categories=allowed_categories,
    )

    headers = assemble_headers(api_key=api_key, usecase_id=usecase_id, app_id=app_id)

    raw_text, raw_resp = call_gemini_openai_style(
        openai_client=openai_client,
        model=model,
        system_text=system_text,
        user_text=user_text,
        headers=headers,
        temperature=0.0,
        max_tokens=400,
    )

    # Never hide RAW output; parsing is best-effort.
    try:
        parsed_inner = parse_classifier_output(raw_text)
        # Snap category to allowed list
        norm_cat, score, is_exact = normalize_category(parsed_inner["category"], allowed_categories)
        parsed = {
            "category": norm_cat,
            "task_notes": parsed_inner["task_notes"],
            "confidence": parsed_inner["confidence"],
            "raw_category": parsed_inner["category"],
            "category_match_score": score,
            "category_exact_match": is_exact,
        }
    except Exception as e:
        parsed = {
            "parse_error": str(e),
            "note": "Parser could not interpret model output; see raw_text for full content.",
        }

    return {
        "raw_text": raw_text,
        "parsed": parsed,
        "request_headers_used": headers,
    }


# ---------- Helpers ----------

def parse_allowed_categories(s: str) -> List[str]:
    """
    Parse a comma-separated list of allowed categories into a clean list.
    Example input:
      "KYC UPDATE, ADDRESS CHANGE, DDA MAINTENANCE"
    """
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(description="Case email classifier via Gemini (OpenAI-style, Apigee).")

    parser.add_argument("--examples-csv", required=True, help="Path to CSV of labeled examples.")
    parser.add_argument("--email-col", default="email_text", help="Column for email text in the examples CSV.")
    parser.add_argument("--category-col", default="category", help="Column for category in the examples CSV.")
    parser.add_argument("--notes-col", default="task_notes", help="Column for task notes in the examples CSV.")
    parser.add_argument("--max-examples", type=int, default=5, help="Max number of few-shot examples to use.")

    parser.add_argument("--email-text", required=True, help="Incoming email text to classify.")
    parser.add_argument("--model", default="gemini-2.5.pro", help="Model name.")

    parser.add_argument(
        "--allowed-categories",
        required=True,
        help="Comma-separated list of allowed categories, e.g. "
             "\"KYC UPDATE, ADDRESS CHANGE, DDA MAINTENANCE\"",
    )

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

    allowed_categories = parse_allowed_categories(args.allowed_categories)
    if not allowed_categories:
        raise SystemExit("No allowed categories parsed from --allowed-categories.")

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
    few_shot_block = build_few_shot_block(
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
        allowed_categories=allowed_categories,
    )

    print("\n=== RAW MODEL TEXT ===\n")
    print(result["raw_text"])
    print("\n=== PARSED STRUCTURED OUTPUT (category snapped to allowed list) ===\n")
    print(json.dumps(result["parsed"], indent=2, ensure_ascii=False))
    print("\n=== REQUEST HEADER SUMMARY (no bearer) ===\n")
    print(json.dumps(result["request_headers_used"], indent=2))

if __name__ == "__main__":
    main()

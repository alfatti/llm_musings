# account_sanitizer.py

import re
import json
from typing import List

# -------------------------------------------------------------------
# 1. Basic numeric heuristic (safety net)
# -------------------------------------------------------------------
BASIC_ACCOUNT_REGEX = re.compile(
    r'\b(?:\d{8,16}|\d{3,4}(?:[- ]\d{3,4}){1,4})\b'
)

def _mask_basic_numeric_patterns(text: str, mask_token: str = "***MASKED***") -> str:
    return BASIC_ACCOUNT_REGEX.sub(mask_token, text)


# -------------------------------------------------------------------
# 2. LLM-based account "NER"
# -------------------------------------------------------------------
LLM_SYSTEM_PROMPT = """\
You are an information extraction assistant for a bank.
Your job is to find all substrings in the text that look like ACCOUNT NUMBERS.

Be conservative but slightly over-inclusive. 
Examples of what to include:
- Long numeric IDs (e.g. 12345678, 9876543210123456)
- Chunked forms like 1234-5678-9012, 12 3456 7890 12
- Alphanumeric IDs that are explicitly referred to as accounts
  (e.g. "Account AB12345-09", "acct: XYZ-998877")

Do NOT include:
- Phone numbers
- Zip codes
- Dates
- Monetary amounts
- Short numbers with no account context

Return ONLY a JSON object of the form:
{"account_numbers": ["substring1", "substring2", ...]}
"""

LLM_USER_PROMPT_TEMPLATE = """\
Identify all account numbers in the following text. 
Return them EXACTLY as they appear in the text.

TEXT:
----------------
{}
----------------
"""


def extract_account_candidates_llm(
    text: str,
    client,
    model: str = "gpt-4.1-mini"
) -> List[str]:
    """
    Use an LLM to detect account-like substrings.

    Parameters
    ----------
    text : str
        The email text.
    client : OpenAI client (already initialized)
    model : str
        Model name.

    Returns
    -------
    List[str]
        List of substrings that should be masked.
    """
    user_prompt = LLM_USER_PROMPT_TEMPLATE.format(text)

    # Adjust this to your exact client style if needed
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )

    raw = resp.output[0].content[0].text  # adapt if your client shape differs
    try:
        data = json.loads(raw)
        accounts = data.get("account_numbers", [])
        # normalize to strings and dedupe while keeping order
        seen = set()
        result = []
        for a in accounts:
            s = str(a).strip()
            if s and s not in seen:
                seen.add(s)
                result.append(s)
        return result
    except Exception:
        # If JSON parsing fails, just give up on LLM layer and return empty
        return []


# -------------------------------------------------------------------
# 3. Regex masking over LLM candidates
# -------------------------------------------------------------------
def mask_with_patterns(
    text: str,
    patterns: List[str],
    mask_token: str = "***MASKED***"
) -> str:
    """
    For each pattern string, escape it and mask all occurrences in the text.
    """
    masked = text
    for p in patterns:
        # Escape to treat literal characters (like +, ?, etc.) safely
        escaped = re.escape(p)
        masked = re.sub(escaped, mask_token, masked)
    return masked


# -------------------------------------------------------------------
# 4. Full pipeline
# -------------------------------------------------------------------
def sanitize_email_accounts(
    text: str,
    client,
    model: str = "gpt-4.1-mini",
    mask_token: str = "***MASKED***"
) -> str:
    """
    1) Ask LLM to do a quick-and-dirty NER for account numbers.
    2) Mask all those substrings via regex.
    3) Also run a basic numeric regex as a fallback.

    Returns the sanitized email text.
    """
    # LLM layer
    candidates = extract_account_candidates_llm(text, client=client, model=model)

    masked = text
    if candidates:
        masked = mask_with_patterns(masked, candidates, mask_token=mask_token)

    # Numeric safety net
    masked = _mask_basic_numeric_patterns(masked, mask_token=mask_token)

    return masked


if __name__ == "__main__":
    # Example usage (assumes you've initialized `client` elsewhere)
    from openai import OpenAI
    client = OpenAI()

    sample_email = """
    Hi team,

    Please open a ticket for client ACME.
    Primary account 123456789012 should be used.
    Also reference "Account AB12345-09" and backup 1234-5678-9012.
    Phone 212-555-1234 can remain.

    Thanks.
    """

    clean = sanitize_email_accounts(sample_email, client=client)
    print(clean)

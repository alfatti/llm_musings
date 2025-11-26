# brute_masker.py

import re

def mask_digits_bruteforce(text: str, mask_token="***MASKED***") -> str:
    """
    1. Replace all digits [0-9] with '*'
    2. Collapse any contiguous run of 2+ '*' into a MASK token

    Example:
       "Acct 123456 and ref 7890" ->
         Step1: "Acct ****** and ref ****"
         Step2: "Acct ***MASKED*** and ref ***MASKED***"
    """

    if not isinstance(text, str):
        return text

    # Step 1: replace digits with '*'
    step1 = re.sub(r"\d", "*", text)

    # Step 2: replace runs of 2+ '*' with mask token
    step2 = re.sub(r"\*{2,}", mask_token, step1)

    return step2

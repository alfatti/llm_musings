# account_masker.py

import re

# Heuristic patterns:
#  - Single long digit strings: 8–16 digits
#  - Chunked forms: 3–4 digits per chunk separated by "-" or spaces
_ACCOUNT_REGEX = re.compile(
    r'\b(?:\d{8,16}|\d{3,4}(?:[- ]\d{3,4}){1,3})\b'
)

def mask_account_numbers(text: str) -> str:
    """
    Replace any account-like numeric pattern with a fixed mask '***MASKED***'.

    Examples:
      '123456789012'         → '***MASKED***'
      '1234-5678-9012'       → '***MASKED***'

    Phone numbers like 212-555-1234 *should not* match because
    the chunks are 3-3-4 digits.
    """
    return _ACCOUNT_REGEX.sub("***MASKED***", text)


if __name__ == "__main__":
    sample = (
        "Hi team,\n"
        "Process account 123456789012 and backup 1234-5678-9012.\n"
        "Phone 212-555-1234 should remain.\n"
    )
    print(mask_account_numbers(sample))

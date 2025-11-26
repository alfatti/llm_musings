# account_masker.py

import re

# Very simple heuristic patterns:
#  - Long digit strings: 8–16 digits (e.g. 12345678, 987654321012)
#  - Chunked with dashes/spaces: 3–4 digits per chunk (e.g. 123-456-789, 1234-5678-9012)
_ACCOUNT_REGEX = re.compile(
    r'\b(?:\d{8,16}|\d{3,4}(?:[- ]\d{3,4}){1,3})\b'
)


def _mask_account_like_string(raw: str) -> str:
    """
    Mask all but the last 4 digits in a string that contains digits
    (preserving separators like '-' or spaces).

    Example:
      '123456789012'      -> '********9012'
      '1234-5678-9012'    -> '****-****-9012'
    """
    digits_only = re.sub(r'\D', '', raw)

    # If very short (e.g. '1234'), don't treat as account number
    if len(digits_only) <= 4:
        return raw

    masked_digits = '*' * (len(digits_only) - 4) + digits_only[-4:]

    # Rebuild string, preserving non-digit characters (dashes, spaces)
    out = []
    i = 0
    for ch in raw:
        if ch.isdigit():
            out.append(masked_digits[i])
            i += 1
        else:
            out.append(ch)
    return ''.join(out)


def mask_account_numbers(text: str) -> str:
    """
    Find account-like numeric patterns in the text and mask them.

    Heuristics (on purpose, not perfect):
      - Sequences of 8–16 digits (e.g. '12345678', '1234567890123456')
      - Sequences of 3–4 digit chunks separated by '-' or space
        (e.g. '123-456-789', '1234 5678 9012')

    Masking rule:
      - Keep last 4 digits, replace the rest with '*'
        (format preserved: dashes/spaces stay where they are)

    Example
    -------
    >>> s = "Acct 123456789012 for client, and 9876-5432-1098 as backup."
    >>> mask_account_numbers(s)
    'Acct ********9012 for client, and ****-****-1098 as backup.'
    """
    def repl(match: re.Match) -> str:
        raw = match.group(0)
        return _mask_account_like_string(raw)

    return _ACCOUNT_REGEX.sub(repl, text)


if __name__ == "__main__":
    # Tiny sanity check
    sample = (
        "Hi team,\n\n"
        "Please process for account 123456789012 and backup 1234-5678-9012.\n"
        "Client phone 212-555-1234 should stay as-is.\n"
    )
    print(mask_account_numbers(sample))

import re
import json

def clean_json_string(s: str) -> str:
    """
    Removes code fences like ```json or ``` from a string
    and returns a cleaned JSON string.
    """
    # Remove ```json, ```python, ``` etc.
    s = re.sub(r"```[\w]*", "", s)
    s = s.replace("```", "")
    return s.strip()


# Example usage:
raw = """```json
{"accounts":[12234445, 43216]}
```"""

cleaned = clean_json_string(raw)
data = json.loads(cleaned)

print(data)

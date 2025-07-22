import json
import pandas as pd

def clean_gemini_json_output(raw_output: str):
    lines = raw_output.strip().splitlines()

    # Filter out any lines that are just markdown code fences
    filtered_lines = [
        line for line in lines if not line.strip().startswith("```")
    ]

    return "\n".join(filtered_lines)

# Example input
gemini_output = """
```json
[
  {"Unit": "AAA", "Unit Type": "Studio", "Move Out": null},
  {"Unit": "BBB", "Unit Type": "1BR", "Move Out": "2025-09-01"}
]

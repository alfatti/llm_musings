import json
import pandas as pd

def clean_gemini_json_output(raw_output: str):
    # Remove markdown code block if present
    if raw_output.strip().startswith("```"):
        lines = raw_output.strip().splitlines()
        # Remove first and last line
        lines = lines[1:-1]
        raw_output = "\n".join(lines)
    return raw_output


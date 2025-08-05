import base64
import requests

prompt_text = """You are an expert in cleaning and standardizing messy Excel headers using Pandas.

Given an image of the top rows of an Excel sheet, your task is to:
- Remove non-data rows such as titles, subtitles, or summary rows.
- If column names are split across multiple rows, merge them into single-line headers.
- Output a clean Pandas script that reads the Excel sheet and returns a normalized DataFrame.

Here are several examples. Pay attention to how different messy headers result in clean, consistent outputs.
"""
import base64

def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def build_request_body(prompt_text, example_pairs, target_image_path):
    parts = [{"text": prompt_text}]
    
    for i, (image_path, code_str) in enumerate(example_pairs, 1):
        img_b64 = image_to_base64(image_path)
        parts.extend([
            {"text": f"\nExample {i}:"},
            {"inline_data": {
                "mime_type": "image/png",
                "data": img_b64
            }},
            {"text": f"Output code:\n```python\n{code_str.strip()}\n```"}
        ])

    # Add the target image to prompt generation
    target_b64 = image_to_base64(target_image_path)
    parts.extend([
        {"text": "\nNow write the Pandas code to clean the header of this Excel sheet:"},
        {"inline_data": {
            "mime_type": "image/png",
            "data": target_b64
        }}
    ])

    return {"contents": [{"parts": parts}]}

def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def build_request_body(prompt_text, example_pairs, target_image_path):
    parts = [{"text": prompt_text}]
    
    for i, (image_path, code_str) in enumerate(example_pairs, 1):
        img_b64 = image_to_base64(image_path)
        parts.extend([
            {"text": f"\nExample {i}:"},
            {"inline_data": {
                "mime_type": "image/png",
                "data": img_b64
            }},
            {"text": f"Output code:\n```python\n{code_str.strip()}\n```"}
        ])

    # Add the target image to prompt generation
    target_b64 = image_to_base64(target_image_path)
    parts.extend([
        {"text": "\nNow write the Pandas code to clean the header of this Excel sheet:"},
        {"inline_data": {
            "mime_type": "image/png",
            "data": target_b64
        }}
    ])

    return {"contents": [{"parts": parts}]}

API_URL = "https://YOUR_GEMINI_ENDPOINT"
HEADERS = {
    "Authorization": f"Bearer {YOUR_API_KEY}",
    "Content-Type": "application/json"
}

# Build the request body with prompt, examples, and target image
request_body = build_request_body(prompt_text, example_pairs, target_image_path)

response = requests.post(API_URL, headers=HEADERS, json=request_body)
reply = response.json()
generated_code = reply["candidates"][0]["content"]["parts"][0]["text"]

print(generated_code)

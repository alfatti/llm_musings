# 1️⃣ Define your allowed categories (Python side)
ALLOWED_CATEGORIES = [
    "Account Opening",
    "Signature Maintenance",
    "Representative Update",
    "Treasury / ACH Issue",
    "Credit Line Inquiry",
    "KYC / Documentation Update",
    "Access / Portal Issue",
    "Other",
]


# 2️⃣ Build the JSON schema with an enum for category
# This is the key piece: notice "enum": ALLOWED_CATEGORIES.
case_output_schema = {
    "name": "case_output",
    "schema": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": ALLOWED_CATEGORIES,  # 👈 hard constraint
            },
            "confidence": {
                "type": "number",
            },
            "task_notes": {
                "type": "string",
            },
        },
        "required": ["category", "confidence", "task_notes"],
        "additionalProperties": False,
    },
}

# 3️⃣ Use it in response_format in client.responses.create
from openai import OpenAI
import json

client = OpenAI()

ALLOWED_CATEGORIES = [
    "Account Opening",
    "Signature Maintenance",
    "Representative Update",
    "Treasury / ACH Issue",
    "Credit Line Inquiry",
    "KYC / Documentation Update",
    "Access / Portal Issue",
    "Other",
]

case_output_schema = {
    "name": "case_output",
    "schema": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": ALLOWED_CATEGORIES,
            },
            "confidence": {
                "type": "number",
            },
            "task_notes": {
                "type": "string",
            },
        },
        "required": ["category", "confidence", "task_notes"],
        "additionalProperties": False,
    },
}

prompt = """
Classify this email into exactly one allowed category and write task notes.

ALLOWED CATEGORIES:
- Account Opening
- Signature Maintenance
- Representative Update
- Treasury / ACH Issue
- Credit Line Inquiry
- KYC / Documentation Update
- Access / Portal Issue
- Other

Email:
\"\"\"Hi, we need to update the authorized signers on our operating account...\"\"\"
"""

response = client.responses.create(
    model="google/gemini-2.5-pro",
    input=[
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": prompt,
                }
            ],
        }
    ],
    response_format={
        "type": "json_schema",
        "json_schema": case_output_schema,
    },
)

# Extract text JSON
content = response.output[0].content[0].text
json_str = content.value if hasattr(content, "value") else content
parsed = json.loads(json_str)

print(parsed)
# -> {
#   "category": "Signature Maintenance",
#   "confidence": 0.92,
#   "task_notes": "Middle Office should update ..."
# }

# 4️⃣ Plugging into your existing classify_email function
response = client.responses.create(
    model=model,
    input=[ ... ],
    response_format={
        "type": "json_schema",
        "json_schema": case_output_schema,
    },
)

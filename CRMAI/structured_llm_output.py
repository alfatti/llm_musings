import json
from typing import Any, Dict, Optional, Tuple


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


def call_gemini_openai_style(
    openai_client: Any,
    model: str,
    data_url: str,
    system_text: str,
    user_payload: Dict[str, Any],
    headers: Dict[str, str],
    temperature: float = 0.0,
    max_tokens: int = 800,
) -> Tuple[Optional[str], Any]:
    """
    Calls Gemini (OpenAI-style) with a strict JSON schema enforcing
    allowed categories via enum.
    """

    messages = [
        {"role": "system", "content": system_text},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(user_payload, ensure_ascii=False),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                },
            ],
        },
    ]

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "case_classification",
            "schema": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ALLOWED_CATEGORIES,
                        "description": "Must be exactly one of the allowed categories",
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "task_notes": {
                        "type": "string",
                        "description": "Instructions for Middle Office fulfillment",
                    },
                },
                "required": ["category", "confidence", "task_notes"],
                "additionalProperties": False,
            },
        },
    }

    resp = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
        extra_headers=headers,  # SDK supplies auth
    )

    content = (
        resp.choices[0].message.content
        if (resp and getattr(resp, "choices", None))
        else None
    )

    return content, resp

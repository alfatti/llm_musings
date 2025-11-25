system_text = (
    "Respond EXACTLY in this format:\n"
    "CATEGORY: <category>\n"
    "TASK_NOTES: <notes>\n"
    "CONFIDENCE: <float>\n"
)

user_payload = {
    "incoming_email": "Test: please change address",
    "format": "ONLY output the 3 lines. Nothing else."
}

raw = client.chat.completions.create(
    model="gemini-2.5-pro",
    messages=[
        {"role": "system", "content": system_text},
        {"role": "user", "content": [{"type": "text", "text": json.dumps(user_payload)}]}
    ],
    temperature=0
).choices[0].message.content

print(raw)

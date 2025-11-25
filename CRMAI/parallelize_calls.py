import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

client = OpenAI()

# === 1. Your single-call function that RETURNS A DICT ===
def llm_classify_row(row):
    email_text = row["EMAIL_BODY"]

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "Classify and return JSON: {category, notes, conf}."},
                {"role": "user",  "content": email_text},
            ]
        )

        raw = resp.choices[0].message.content

        # assume you already have a JSON parser from earlier work
        parsed = coerce_json_from_text(raw)

        # must return a dict with fixed keys
        return {
            "category": parsed.get("category"),
            "notes": parsed.get("notes"),
            "conf": parsed.get("confidence")
        }

    except Exception as e:
        return {
            "category": None,
            "notes": f"ERROR: {e}",
            "conf": None
        }


# === 2. Parallel wrapper that returns a Series of DICTS aligned with df.index ===
def parallel_apply_llm(df, max_workers=12):
    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(llm_classify_row, row): idx
            for idx, row in df.iterrows()
        }

        for fut in as_completed(futures):
            idx = futures[fut]
            results[idx] = fut.result()

    return pd.Series(results).sort_index()


# Series of dicts
df["AI_result"] = parallel_apply_llm(df)

# Expand dict into columns
expanded = df["AI_result"].apply(pd.Series)

# Merge back
df = pd.concat([df, expanded], axis=1)

# OPTIONAL: drop the original dict column
# df = df.drop(columns=["AI_result"])

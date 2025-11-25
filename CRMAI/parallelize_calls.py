import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

client = OpenAI()

# === 1. Your single-call function ===
def classify_email(row):
    email_text = row["EMAIL_BODY"]

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "Categorize emails into predefined categories."},
                {"role": "user", "content": email_text},
            ]
        )

        out = resp.choices[0].message.content
        return { "index": row.name, "raw": out }

    except Exception as e:
        return { "index": row.name, "error": str(e), "raw": None }


# === 2. Parallel pipeline ===
def parallel_classify(df, max_workers=10):
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(classify_email, row): row.name for _, row in df.iterrows()}

        for fut in as_completed(futures):
            results.append(fut.result())

    # merge results back into df
    results_df = pd.DataFrame(results).set_index("index").sort_index()
    return pd.concat([df, results_df], axis=1)


# === 3. Usage ===
# df must have EMAIL_BODY column
df_out = parallel_classify(df, max_workers=12)
df_out.head()

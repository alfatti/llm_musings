import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

client = OpenAI()

# === 1. Your single-call function (returns ONE VALUE for the new column) ===
def llm_classify_row(row):
    email_text = row["EMAIL_BODY"]

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "Categorize the email."},
                {"role": "user", "content": email_text},
            ]
        )

        return resp.choices[0].message.content  # <- this will go into AI_result

    except Exception as e:
        return f"ERROR: {str(e)}"


# === 2. Parallel wrapper that returns a Series aligned with df.index ===
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

    # return a Pandas Series aligned with df
    return pd.Series(results).sort_index()


# === 3. Use it ===
df["AI_result"] = parallel_apply_llm(df)

import json
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional

MAX_ITER = 3  # Max retry attempts per page

def extract_rentroll_page(page_text: str) -> Dict[str, Any]:
    """
    Your actual LLM extraction function here. Must return a JSON-serializable object.
    Replace this with your real model logic.
    """
    raise NotImplementedError("Replace with actual LLM call to extract rent roll.")

def try_convert_to_df(json_obj: Any, page_idx: int) -> Tuple[Optional[pd.DataFrame], bool]:
    """
    Try converting JSON object to DataFrame.
    Returns (DataFrame, success_flag).
    """
    try:
        df = pd.DataFrame(json_obj)
        return df, True
    except Exception as e:
        print(f"[Page {page_idx}] ❌ DataFrame conversion failed: {e}")
        return None, False

def self_correcting_rentroll_extractor(page_texts: List[str]) -> Tuple[List[pd.DataFrame], List[int]]:
    """
    Runs extraction with retry for failed pages. Returns:
    - list of successfully extracted DataFrames
    - list of page indices that failed even after retries
    """
    num_pages = len(page_texts)
    json_outputs = [None] * num_pages
    dataframes = [None] * num_pages
    error_page_nums = list(range(num_pages))  # assume all pages need extraction initially

    for iteration in range(1, MAX_ITER + 1):
        print(f"\n🔁 Iteration {iteration} | Retrying {len(error_page_nums)} pages")

        new_errors = []

        for idx in error_page_nums:
            try:
                print(f"→ [Page {idx}] extracting...")
                json_obj = extract_rentroll_page(page_texts[idx])
                json_outputs[idx] = json_obj

                df, success = try_convert_to_df(json_obj, idx)
                if success:
                    dataframes[idx] = df
                else:
                    new_errors.append(idx)

            except Exception as e:
                print(f"[Page {idx}] ❌ Extraction failed: {e}")
                new_errors.append(idx)

        if not new_errors:
            print("✅ All pages processed successfully.")
            break

        error_page_nums = new_errors

    # Collect successful DataFrames
    final_dataframes = [df for df in dataframes if df is not None]
    remaining_errors = error_page_nums

    if remaining_errors:
        print(f"\n⚠️ Final unresolved pages: {remaining_errors}")

    return final_dataframes, remaining_errors

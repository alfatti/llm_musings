import pandas as pd
import random
from typing import List, Dict, Optional


def few_shot_prompt_builder(
    examples_csv_path: str,
    user_email: str,
    sample_size: int = 10,
    allowed_categories: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Build a fully structured prompt for the OpenAI client().chat.completions.create() call.
    
    Parameters
    ----------
    examples_csv_path : str
        Path to CSV containing columns: ['email', 'category', 'task_notes']
        
    user_email : str
        The incoming client email to be evaluated by the LLM.
        
    sample_size : int
        Number of few-shot examples to randomly sample for the prompt.
        
    allowed_categories : List[str], optional
        A fixed list of allowed categories. If None, categories will be inferred from the CSV.

    Returns
    -------
    List[Dict]
        A structured prompt compatible with:
        client.chat.completions.create(model=..., messages=prompt)
    """

    # Load dataset
    df = pd.read_csv(examples_csv_path)

    required_cols = {"email", "category", "task_notes"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    # Define categories (unless provided)
    if allowed_categories is None:
        allowed_categories = sorted(df["category"].unique().tolist())

    # Random sampling for few-shot examples
    if sample_size > len(df):
        sample_size = len(df)
    sampled = df.sample(sample_size, random_state=None)

    # Build few-shot examples block
    example_blocks = []
    for _, row in sampled.iterrows():
        example_block = f"""
### EXAMPLE
EMAIL:
{row['email']}

CATEGORY:
{row['category']}

TASK_NOTES:
{row['task_notes']}
"""
        example_blocks.append(example_block)

    few_shot_section = "\n\n".join(example_blocks)

    # SYSTEM + FEW-SHOT + USER message triplet
    system_prompt = f"""
You are an expert Corporate Client Service Analyst.

You receive emails from clients and must:
1. Assign exactly one category from the allowed list.
2. Write professional, concise task notes for the Middle Office.
3. Estimate your confidence (0.00–1.00).
4. Return strictly valid JSON only.

ALLOWED CATEGORIES:
{allowed_categories}

OUTPUT JSON FORMAT (strict):
{{
   "category": "...",
   "confidence": 0.00,
   "task_notes": "..."
}}

FEW-SHOT TRAINING EXAMPLES:
{few_shot_section}

Now classify the USER EMAIL below.
"""

    user_prompt = f"""USER EMAIL:
{user_email}
"""

    # Output in OpenAI client() message format
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return messages

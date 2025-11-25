import json
from typing import Any, Dict, Optional

import pandas as pd
from openai import OpenAI

client = OpenAI()  # assume you've set OPENAI_API_KEY etc.


JUDGE_PROMPT_TEMPLATE = """
You are an expert reviewer for a client-servicing workflow at a large bank.

TASK
You will compare:
1) The HUMAN_GROUND_TRUTH notes (the notes that the Relationship Associate actually entered into the case-management system).
2) The AI_PROPOSED notes (what the AI assistant suggested).

Your job is to grade how good the AI_PROPOSED notes are as a substitute for the HUMAN_GROUND_TRUTH notes, and then explain your grade and suggest an improved version.

CONTEXT
- The notes are used to route a client email to the correct middle-office team and to give that team clear instructions on what to do next.
- Good notes should:
  - Be factually consistent with the client’s email and the human notes.
  - Capture the key intent and required actions (what must be done, by whom, for which account/relationship).
  - Include any critical details needed for processing (identifiers, accounts, dates, products, amounts) when present in the human notes.
  - Use a concise, professional tone suitable for internal case-routing.

EVALUATION DIMENSIONS
Score each dimension from 0.0 to 1.0:
- correctness: Are the AI notes free of hallucinations and factual errors relative to the human notes?
- completeness: Do the AI notes capture all critical actions and details present in the human notes?
- actionability: Would the middle-office team know exactly what to do from reading the AI notes alone?
- tone_and_professionalism: Are the AI notes concise, neutral, and professional?

OVERALL GRADE
- grade_numeric: A continuous score from 0.0 (unusable) to 1.0 (as good as or better than the human notes).
- grade_label (based on grade_numeric):
  - EXCELLENT: grade_numeric >= 0.9
  - GOOD: 0.75 <= grade_numeric < 0.9
  - FAIR: 0.5 <= grade_numeric < 0.75
  - POOR: 0.25 <= grade_numeric < 0.5
  - UNUSABLE: grade_numeric < 0.25

INSTRUCTIONS
1) Carefully read HUMAN_GROUND_TRUTH and AI_PROPOSED.
2) Identify:
   - What the human notes say the case is about.
   - The concrete actions requested, including any key identifiers.
3) Compare the AI_PROPOSED notes against the human notes on the four dimensions.
4) Assign the numeric scores and grade_label following the rules above.
5) Explain briefly the most important differences (missing actions, extra/hallucinated details, ambiguity, etc.).
6) Rewrite the AI notes into an improved version that would deserve an EXCELLENT grade, staying faithful to the human notes.

IMPORTANT
- Do NOT invent details that are not present in either set of notes.
- If the AI notes mention details that the human notes do not, treat them as potential hallucinations and penalize correctness unless they are clearly benign paraphrases.
- Your final output MUST be valid JSON only, with no extra text, following exactly this schema:

{
  "grade_numeric": <float between 0.0 and 1.0>,
  "grade_label": "<EXCELLENT|GOOD|FAIR|POOR|UNUSABLE>",
  "dimension_scores": {
    "correctness": <float between 0.0 and 1.0>,
    "completeness": <float between 0.0 and 1.0>,
    "actionability": <float between 0.0 and 1.0>,
    "tone_and_professionalism": <float between 0.0 and 1.0>
  },
  "key_differences": [
    "<string difference 1>",
    "<string difference 2>"
  ],
  "verdict_explanation": "<short paragraph>",
   "improved_ai_notes": "<rewritten notes in concise professional style>"
}

NOW HERE ARE THE NOTES TO COMPARE:

HUMAN_GROUND_TRUTH:
<<<HUMAN_NOTES_START>>>
{human_notes}
<<<HUMAN_NOTES_END>>>

AI_PROPOSED:
<<<AI_NOTES_START>>>
{ai_notes}
<<<AI_NOTES_END>>>
""".strip()


def build_judge_prompt(human_notes: str, ai_notes: str) -> str:
    return JUDGE_PROMPT_TEMPLATE.format(
        human_notes=human_notes or "",
        ai_notes=ai_notes or "",
    )


def coerce_json_from_text(raw_text: str) -> Dict[str, Any]:
    """
    Try to extract a JSON object from the model output.
    Assumes there's at least one {...} block; takes the outermost.
    """
    raw_text = raw_text.strip()
    # fast path
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    # fallback: grab substring between first '{' and last '}'
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Could not find JSON object in text:\n{raw_text[:400]}")

    candidate = raw_text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Could not parse JSON from model output: {e}\nRAW (truncated):\n{raw_text[:400]}"
        )


def call_llm_judge_single(
    human_notes: str,
    ai_notes: str,
    model: str = "gpt-4.1-mini",  # replace with your Gemini model wired through OpenAI
    return_raw: bool = False,
) -> Dict[str, Any]:
    """
    Call the LLM-as-judge for a single (human_notes, ai_notes) pair.
    Returns the parsed dict. If return_raw=True, also includes 'raw_text'.
    """
    prompt = build_judge_prompt(human_notes, ai_notes)

    resp = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )

    # Adjust this indexing if your client wrapper differs
    raw_text = resp.output[0].content[0].text

    parsed = coerce_json_from_text(raw_text)

    if return_raw:
        parsed = {**parsed, "raw_text": raw_text}
    return parsed


def evaluate_notes_dataframe(
    df: pd.DataFrame,
    human_col: str = "human_notes",
    ai_col: str = "ai_notes",
    model: str = "gpt-4.1-mini",
    prefix: str = "judge_",
) -> pd.DataFrame:
    """
    Apply the LLM judge row-wise on a sample dataset.

    Expects df[human_col] and df[ai_col] to contain the notes.
    Appends columns like:
        judge_grade_numeric
        judge_grade_label
        judge_correctness
        judge_completeness
        judge_actionability
        judge_tone_and_professionalism
        judge_verdict_explanation
        judge_improved_ai_notes
    """

    def _eval_row(row) -> Optional[Dict[str, Any]]:
        human = row.get(human_col, "")
        ai = row.get(ai_col, "")
        if not human and not ai:
            return None
        try:
            return call_llm_judge_single(human, ai, model=model)
        except Exception as e:
            # You can log e here if you want
            return {
                "grade_numeric": None,
                "grade_label": None,
                "dimension_scores": {
                    "correctness": None,
                    "completeness": None,
                    "actionability": None,
                    "tone_and_professionalism": None,
                },
                "key_differences": [f"LLM call failed: {e}"],
                "verdict_explanation": "",
                "improved_ai_notes": "",
            }

    results = df.apply(_eval_row, axis=1)

    # explode the dicts into columns
    df[f"{prefix}grade_numeric"] = results.apply(
        lambda r: None if r is None else r.get("grade_numeric")
    )
    df[f"{prefix}grade_label"] = results.apply(
        lambda r: None if r is None else r.get("grade_label")
    )

    df[f"{prefix}correctness"] = results.apply(
        lambda r: None if r is None else r.get("dimension_scores", {}).get("correctness")
    )
    df[f"{prefix}completeness"] = results.apply(
        lambda r: None if r is None else r.get("dimension_scores", {}).get("completeness")
    )
    df[f"{prefix}actionability"] = results.apply(
        lambda r: None if r is None else r.get("dimension_scores", {}).get("actionability")
    )
    df[f"{prefix}tone_and_professionalism"] = results.apply(
        lambda r: None
        if r is None
        else r.get("dimension_scores", {}).get("tone_and_professionalism")
    )

    df[f"{prefix}key_differences"] = results.apply(
        lambda r: None if r is None else r.get("key_differences")
    )
    df[f"{prefix}verdict_explanation"] = results.apply(
        lambda r: None if r is None else r.get("verdict_explanation")
    )
    df[f"{prefix}improved_ai_notes"] = results.apply(
        lambda r: None if r is None else r.get("improved_ai_notes")
    )

    return df

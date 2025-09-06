# %% [markdown]
# Few-shot VLM pipeline for PDF rent-roll extraction with Gemini (Vertex AI REST)
# - Builds few-shot examples from (1-page PDF, Excel gold) pairs
# - Shreds test PDFs into page images
# - Parallelizes page calls with ThreadPoolExecutor
# - Ensures JSON-only outputs
# - Minimal changes vs your earlier zero-shot REST pipeline

# %%
import base64
import io
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
from PIL import Image

# pdf2image requires poppler. On macOS: `brew install poppler`
# If not installed, you'll get an ImportError here.
from pdf2image import convert_from_path


# -----------------------------
# Configuration (adjust to your env)
# -----------------------------
PROJECT_ID   = "YOUR_PROJECT_ID"
LOCATION     = "us-central1"   # e.g. "us-central1", "europe-west4"
MODEL_ID     = "publishers/google/models/gemini-1.5-pro"  # or your specific model
ACCESS_TOKEN = "YOUR_OAUTH2_BEARER_TOKEN"  # supply fresh token; do not hardcode in prod

# Optional: global generation config for JSON-only responses
GEN_CONFIG = {
    "temperature": 0.2,
    "candidateCount": 1,
    "maxOutputTokens": 2048,
    "stopSequences": [],
    # Enables better JSON compliance in newer models. Safe if unsupported.
    "responseMimeType": "application/json",
}

SAFETY_SETTINGS = [
    # Leave empty or configure as needed; keeping minimal here to avoid blocking
]


# -----------------------------
# Utilities
# -----------------------------
def build_vertex_endpoint(project_id: str, location: str, model_id: str) -> str:
    """
    Returns the Vertex AI Generative endpoint for REST.
    Example:
    https://us-central1-aiplatform.googleapis.com/v1/projects/xxx/locations/us-central1/publishers/google/models/gemini-1.5-pro:generateContent
    """
    return (
        f"https://{location}-aiplatform.googleapis.com/v1/projects/"
        f"{project_id}/locations/{location}/{model_id}:generateContent"
    )


def pil_to_base64_png(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def pdf_to_pil_images(pdf_path: str, dpi: int = 200) -> List[Image.Image]:
    """
    Converts a (possibly multi-page) PDF to a list of PIL images, one per page.
    Keeps images in memory.
    """
    images: List[Image.Image] = convert_from_path(pdf_path, dpi=dpi)
    return images


def shred_pdf_to_images(pdf_path: str, dpi: int = 200) -> List[Image.Image]:
    """Alias for clarity."""
    return pdf_to_pil_images(pdf_path, dpi=dpi)


def read_excel_gold_to_json(excel_path: str,
                            sheet_name: Optional[str] = None,
                            orient: str = "records",
                            keep_default_na: bool = False) -> str:
    """
    Reads a gold-standard Excel extract and converts to JSON string (list-of-dicts by default).
    - `orient="records"` yields `[{"col": val, ...}, ...]`
    - Sets keep_default_na=False so empty cells become "" rather than NaN (nicer for JSON)
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name, dtype=object, keep_default_na=keep_default_na)
    # If the gold has multiple sheets, you can customize or concat; here we take the first if name not provided.
    if isinstance(df, dict):
        # pick the first sheet if multiple returned
        first_key = list(df.keys())[0]
        df = df[first_key]
    # Fill NaNs consistently
    df = df.fillna("")
    return df.to_json(orient=orient, force_ascii=False)


def extract_json_block(text: str) -> str:
    """
    Robustly extract the first JSON object or array from a string.
    - Handles replies that include preambles or code fences.
    - Returns a compact JSON string; raises ValueError if no JSON found.
    """
    # Strip code fences if present
    fenced = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, flags=re.DOTALL)
    if fenced:
        payload = fenced.group(1)
        return json.dumps(json.loads(payload))  # normalize/compact

    # Otherwise, scan for the first {...} or [...]
    # Simple bracket matcher
    def find_balanced(s: str, open_ch: str, close_ch: str) -> Optional[str]:
        start = s.find(open_ch)
        while start != -1:
            depth = 0
            for i in range(start, len(s)):
                if s[i] == open_ch:
                    depth += 1
                elif s[i] == close_ch:
                    depth -= 1
                    if depth == 0:
                        candidate = s[start:i+1]
                        try:
                            json.loads(candidate)
                            return candidate
                        except json.JSONDecodeError:
                            break
            start = s.find(open_ch, start + 1)
        return None

    for (o, c) in [("{", "}"), ("[", "]")]:
        candidate = find_balanced(text, o, c)
        if candidate is not None:
            return json.dumps(json.loads(candidate))

    raise ValueError("No JSON object/array found in the response.")


# -----------------------------
# Few-shot example container
# -----------------------------
@dataclass
class FewShotExample:
    image_b64_png: str     # inline base64 PNG
    gold_json: str         # JSON string (list-of-dicts or object)


def build_examples_from_paths(pdf_paths: List[str], excel_paths: List[str]) -> List[FewShotExample]:
    """
    Given aligned lists of 1-page PDFs and Excel golds, produce few-shot examples.
    Assumes each PDF is exactly 1 page; if not, uses the first page.
    """
    if len(pdf_paths) != len(excel_paths):
        raise ValueError("pdf_paths and excel_paths must be same length.")

    examples: List[FewShotExample] = []
    for pdf_path, xls_path in zip(pdf_paths, excel_paths):
        imgs = pdf_to_pil_images(pdf_path, dpi=200)
        if len(imgs) == 0:
            raise ValueError(f"No pages found in {pdf_path}")
        image_b64 = pil_to_base64_png(imgs[0])
        gold_json = read_excel_gold_to_json(xls_path)
        examples.append(FewShotExample(image_b64_png=image_b64, gold_json=gold_json))
    return examples


# -----------------------------
# Prompt assembly (few-shot VLM)
# -----------------------------
def build_contents_for_fewshot(
    examples: List[FewShotExample],
    system_preamble: str,
    json_contract_note: str = "Return ONLY valid JSON. No prose, no code fences.",
) -> List[Dict[str, Any]]:
    """
    Build Vertex 'contents' with alternating user/model turns for few-shot learning:
      [ {role:user, parts:[TEXT, IMAGE]}, {role:model, parts:[JSON]}, ... , {role:user, parts:[TEXT, IMAGE]} ]
    The final user turn (without a following model turn) is reserved for the actual test page(s), provided later.
    """
    contents: List[Dict[str, Any]] = []

    # Optional system steering as initial user turn text (Gemini doesn't have a dedicated 'system' role on REST).
    if system_preamble:
        contents.append({
            "role": "user",
            "parts": [{"text": system_preamble}]
        })

    # Few-shot pairs
    for ex in examples:
        contents.append({
            "role": "user",
            "parts": [
                {"text": "Here is a rent-roll page image. Extract the table into the target JSON schema."},
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": ex.image_b64_png
                    }
                },
                {"text": json_contract_note}
            ]
        })
        contents.append({
            "role": "model",
            "parts": [{"text": ex.gold_json}]
        })

    # Return so the caller can append the *actual* test image(s) as the final user turn(s).
    return contents


def append_test_image_turn(contents: List[Dict[str, Any]],
                           instruction: str,
                           pil_image: Image.Image,
                           json_contract_note: str = "Return ONLY valid JSON. No prose, no code fences.") -> None:
    """Append a final user turn with the test image to the pre-built contents."""
    contents.append({
        "role": "user",
        "parts": [
            {"text": instruction},
            {"inline_data": {"mime_type": "image/png", "data": pil_to_base64_png(pil_image)}},
            {"text": json_contract_note}
        ]
    })


# -----------------------------
# Gemini REST call (Vertex AI)
# -----------------------------
import requests

def call_gemini_rest(
    contents: List[Dict[str, Any]],
    project_id: str = PROJECT_ID,
    location: str = LOCATION,
    model_id: str = MODEL_ID,
    access_token: str = ACCESS_TOKEN,
    generation_config: Optional[Dict[str, Any]] = None,
    safety_settings: Optional[List[Dict[str, Any]]] = None,
    retries: int = 2,
    timeout_s: int = 60,
) -> Dict[str, Any]:
    """
    Minimal REST call to Vertex AI Gemini 'generateContent'.
    Returns the parsed JSON response (dict). Does not post-process candidate text.
    """
    endpoint = build_vertex_endpoint(project_id, location, model_id)
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json; charset=UTF-8",
    }

    body: Dict[str, Any] = {"contents": contents}
    if generation_config:
        body["generationConfig"] = generation_config
    if safety_settings:
        body["safetySettings"] = safety_settings

    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(endpoint, headers=headers, json=body, timeout=timeout_s)
            if resp.status_code == 200:
                return resp.json()
            else:
                last_err = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
        except Exception as e:
            last_err = e
        time.sleep(0.6 * (attempt + 1))

    raise last_err if last_err else RuntimeError("Unknown error calling Gemini REST.")


def response_text_or_empty(resp: dict) -> str:
    """
    Extracts all 'text' fields from candidate parts.
    Also surfaces safety blocks and common error shapes.
    """
    # Safety / prompt feedback block?
    pf = resp.get("promptFeedback")
    if pf and pf.get("blockReason"):
        reason = pf.get("blockReason")
        details = pf.get("safetyRatings", [])
        raise RuntimeError(f"Prompt blocked by safety. blockReason={reason}; safetyRatings={details}")

    cands = resp.get("candidates") or []
    if not cands:
        # Some older transports use 'predictions'
        preds = resp.get("predictions") or []
        if preds:
            # Try the common text location
            try:
                return preds[0]["content"]["parts"][0]["text"]
            except Exception:
                pass
        # Nothing we can parse—dump a short diagnostic
        raise RuntimeError(f"No candidates in response. Top-level keys: {list(resp.keys())[:10]}")

    parts = (cands[0].get("content") or {}).get("parts") or []
    texts = []
    for p in parts:
        if "text" in p and isinstance(p["text"], str):
            texts.append(p["text"])
    # Join all text parts (some models split output across parts)
    joined = "\n".join(texts).strip()

    if not joined:
        # Last-ditch: sometimes JSON is sent as a functionCall or other part
        # Surface a concise diagnostic to help debugging
        raise RuntimeError(f"Empty text parts. First part keys: {list(parts[0].keys()) if parts else '[]'}; candidates keys: {list(cands[0].keys())}")

    return joined



# -----------------------------
# Parallel page inference
# -----------------------------
def infer_single_page_json(
    page_image: Image.Image,
    fewshot_examples: List[FewShotExample],
    instruction: str,
    system_preamble: str,
    access_token: str = ACCESS_TOKEN,
) -> str:
    contents = build_contents_for_fewshot(
        examples=fewshot_examples,
        system_preamble=system_preamble,
        json_contract_note="Return ONLY valid JSON. No prose, no code fences."
    )
    append_test_image_turn(
    contents,
    instruction=instruction + "\n\nOutput format:\n```json\n[ { ... } ]\n```\nReturn only the JSON block.",
    pil_image=page_image
)

    resp = call_gemini_rest(
        contents=contents,
        access_token=access_token,
        generation_config=GEN_CONFIG,
        safety_settings=SAFETY_SETTINGS,
    )

    # Quick peek if you’re debugging:
    # print(json.dumps(resp, indent=2)[:1500])

    raw = response_text_or_empty(resp)
    return extract_json_block(raw)



def infer_pdf_pages_parallel(
    pdf_path: str,
    fewshot_examples: List[FewShotExample],
    instruction: str,
    system_preamble: str,
    max_workers: int = 4,
) -> Dict[int, str]:
    """
    Shreds a (multi-page) PDF, runs each page in parallel, returns {page_index: json_string}.
    Page indices are 0-based in the order produced by pdf2image.
    """
    images = shred_pdf_to_images(pdf_path, dpi=200)
    outputs: Dict[int, str] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(
                infer_single_page_json,
                img,
                fewshot_examples,
                instruction,
                system_preamble,
                ACCESS_TOKEN,
            ): i
            for i, img in enumerate(images)
        }

        for fut in as_completed(futs):
            i = futs[fut]
            try:
                outputs[i] = fut.result()
            except Exception as e:
                outputs[i] = json.dumps({"error": str(e)})

    return dict(sorted(outputs.items(), key=lambda kv: kv[0]))


# -----------------------------
# Convenience: End-to-end assembly
# -----------------------------
DEFAULT_SYSTEM = (
    "You are a precise data-extraction model for apartment rent-rolls. "
    "Given a single page image, extract rows into a strict JSON schema.\n\n"
    "Schema expectations (example):\n"
    "[\n"
    "  {\n"
    '    "Property": "string",\n'
    '    "Unit": "string",\n'
    '    "Resident": "string",\n'
    '    "Status": "string",\n'
    '    "SQFT": "number|string",\n'
    '    "Lease_From": "YYYY-MM-DD|string",\n'
    '    "Lease_To": "YYYY-MM-DD|string",\n'
    '    "ChargeTypes": {"Rent": number, "Garbage": number, "Pet": number, ...},\n'
    '    "Amount": number,\n'
    '    "...": "..." \n'
    "  }\n"
    "]\n"
    "If a field is missing, put an empty string. Return ONLY JSON."
)

DEFAULT_INSTRUCTION = (
    "Extract the table on this page into the JSON schema described earlier. "
    "Return ONLY valid JSON (no prose)."
)


def build_fewshot_from_pairs(
    pdf_example_paths: List[str],
    excel_gold_paths: List[str],
) -> List[FewShotExample]:
    """Public wrapper for building examples."""
    return build_examples_from_paths(pdf_example_paths, excel_gold_paths)


# -----------------------------
# Example usage (uncomment and edit paths)
# -----------------------------
# pdf_examples = [
#     "examples/rr_example_1.pdf",
#     "examples/rr_example_2.pdf",
#     "examples/rr_example_3.pdf",
#     "examples/rr_example_4.pdf",
#     "examples/rr_example_5.pdf",
#     "examples/rr_example_6.pdf",
# ]
# excel_golds = [
#     "examples/rr_example_1_gold.xlsx",
#     "examples/rr_example_2_gold.xlsx",
#     "examples/rr_example_3_gold.xlsx",
#     "examples/rr_example_4_gold.xlsx",
#     "examples/rr_example_5_gold.xlsx",
#     "examples/rr_example_6_gold.xlsx",
# ]
# fewshot = build_fewshot_from_pairs(pdf_examples, excel_golds)
#
# # Run on a test PDF (1 or many pages). Output: {page_idx: json_string}
# results = infer_pdf_pages_parallel(
#     pdf_path="tests/sample_test.pdf",
#     fewshot_examples=fewshot,
#     instruction=DEFAULT_INSTRUCTION,
#     system_preamble=DEFAULT_SYSTEM,
#     max_workers=4,
# )
# print(json.dumps(results, indent=2))


# %% [markdown]
# Inspector for Few-Shot (image, JSON) examples
# %% [markdown]
# ------- Few-shot context inspector (text + images) -------

import io
import json
import base64
from typing import List, Optional
import matplotlib.pyplot as plt
from PIL import Image

# If not already in your notebook:
# from dataclasses import dataclass
# @dataclass
# class FewShotExample:
#     image_b64_png: str
#     gold_json: str

def _b64_to_pil(b64_png: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_png)))

def inspect_fewshot_context(
    fewshot_examples: List[FewShotExample],
    system_preamble: str,
    per_example_user_text: str = "Here is a rent-roll page image. Extract the table into the target JSON schema.",
    final_test_instruction: str = "Extract the table on this page into the JSON schema described earlier. Return ONLY valid JSON (no prose).",
    show_images: bool = True,
) -> None:
    """
    Prints and renders the full textual context that will be sent to Gemini:
      - System preamble (what we pass as the first 'user' turn)
      - For each few-shot example:
          * user text (the per-example instruction)
          * image (inline)
          * model gold JSON
      - Final test instruction (what will precede the actual test image turn)
    """
    print("="*80)
    print("SYSTEM PREAMBLE (first user turn):")
    print("-"*80)
    print(system_preamble.strip())
    print()

    for i, ex in enumerate(fewshot_examples, 1):
        print("="*80)
        print(f"FEW-SHOT EXAMPLE #{i}")
        print("-"*80)
        print("User text (few-shot instruction):")
        print(per_example_user_text.strip())
        print()

        if show_images:
            im = _b64_to_pil(ex.image_b64_png)
            plt.figure(figsize=(6, 8))
            plt.imshow(im)
            plt.axis("off")
            plt.title(f"Few-shot image #{i}")
            plt.show()

        print("Model gold JSON:")
        try:
            parsed = json.loads(ex.gold_json)
            print(json.dumps(parsed, indent=2, ensure_ascii=False))
        except Exception:
            print(ex.gold_json)
        print()

    print("="*80)
    print("FINAL TEST INSTRUCTION (the text paired with your real test image):")
    print("-"*80)
    print(final_test_instruction.strip())
    print()


def build_contents_preview_without_test_image(
    fewshot_examples: List[FewShotExample],
    system_preamble: str,
    per_example_user_text: str = "Here is a rent-roll page image. Extract the table into the target JSON schema.",
    json_contract_note: str = "Return ONLY valid JSON. No prose, no code fences.",
) -> list:
    """
    Builds the exact 'contents' structure (minus the final test turn) to preview
    what will be POSTed to Gemini. This mirrors your few-shot assembly:
      [ {role:user, parts:[system_preamble]},
        {role:user, parts:[per_example_user_text, image, json_contract_note]},
        {role:model, parts:[gold_json]}, ... ]
    """
    contents = []
    if system_preamble:
        contents.append({"role": "user", "parts": [{"text": system_preamble}]})

    for ex in fewshot_examples:
        contents.append({
            "role": "user",
            "parts": [
                {"text": per_example_user_text},
                {"inline_data": {"mime_type": "image/png", "data": ex.image_b64_png}},
                {"text": json_contract_note},
            ],
        })
        contents.append({
            "role": "model",
            "parts": [{"text": ex.gold_json}],
        })

    return contents


def preview_contents_as_json(
    contents: list,
    truncate_text: Optional[int] = 400,
) -> None:
    """
    Pretty-prints a concise preview of the contents payload.
    Large blobs (images/gold JSON) can be truncated for readability.
    """
    def trunc(s: str) -> str:
        if truncate_text and len(s) > truncate_text:
            return s[:truncate_text] + f"... [truncated {len(s)-truncate_text} chars]"
        return s

    preview = []
    for turn in contents:
        role = turn.get("role", "?")
        parts_preview = []
        for p in turn.get("parts", []):
            if "text" in p:
                parts_preview.append({"text": trunc(p["text"])})
            elif "inline_data" in p:
                meta = p["inline_data"]
                parts_preview.append({
                    "inline_data": {
                        "mime_type": meta.get("mime_type", "image/png"),
                        "data": f"<base64 {len(meta.get('data',''))} chars>",
                    }
                })
            else:
                parts_preview.append({"unknown_part": True})
        preview.append({"role": role, "parts": parts_preview})

    print(json.dumps(preview, indent=2, ensure_ascii=False))


# ---------------------------
# Example usage (after you build `fewshot` + have DEFAULT_SYSTEM/DEFAULT_INSTRUCTION):
#
# inspect_fewshot_context(
#     fewshot_examples=fewshot,
#     system_preamble=DEFAULT_SYSTEM,
#     per_example_user_text="Here is a rent-roll page image. Extract the table into the target JSON schema.",
#     final_test_instruction=DEFAULT_INSTRUCTION,
#     show_images=True,
# )
#
# contents_preview = build_contents_preview_without_test_image(
#     fewshot_examples=fewshot,
#     system_preamble=DEFAULT_SYSTEM,
#     per_example_user_text="Here is a rent-roll page image. Extract the table into the target JSON schema.",
# )
# preview_contents_as_json(contents_preview, truncate_text=500)
# ---------------------------




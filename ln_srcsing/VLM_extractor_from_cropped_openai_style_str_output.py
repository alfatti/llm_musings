#1) Replace your build_prompt(...) with a text-mode version

def build_prompt(
    variable_name: str,
    section_variants: List[str],
    subsection_variants: List[str],
    line_item_variants: List[str],
    cell_hint: Optional[str],
    vicinity_rows: int,
    vicinity_cols: int,
) -> Tuple[str, Dict[str, Any]]:
    col_letters, hint_row, hint_col_idx = parse_cell(cell_hint) if cell_hint else (None, None, None)

    vicinity_desc: Dict[str, Any] = {
        "bias": "vertical-first",
        "rows_to_check_above": vicinity_rows,
        "rows_to_check_below": vicinity_rows,
        "cols_to_check_left": vicinity_cols,
        "cols_to_check_right": vicinity_cols,
        "cell_hint": cell_hint or "UNKNOWN",
        "parsed_hint": {"column_letters": col_letters, "row_index": hint_row, "column_index_1_based": hint_col_idx},
    }

    system_text = (
        "You are a precise financial document vision assistant. "
        "Inspect a single-page spreadsheet image (exported from PDF) and return exactly one line of TEXT. "
        "No markdown, no JSON, no additional words."
    )

    # We still pass structured guidance as JSON text to improve accuracy,
    # but the REQUIRED output is a single pipe-delimited line.
    user_text: Dict[str, Any] = {
        "task": "Extract the variable: '{}'.".format(variable_name),
        "name_pointers": {
            "section_variants": section_variants,
            "subsection_variants": subsection_variants,
            "line_item_variants": line_item_variants,
            "notes": "Variants come from prior reports; use fuzzy matching and layout cues.",
        },
        "cell_hint_and_vicinity": vicinity_desc,
        "layout_prior": (
            "Rows usually correspond to loans and can shift between versions. "
            "Bias vertical scanning near the hint row; then sweep limited columns horizontally."
        ),
        "value_rules": [
            "Return the value exactly as seen (verbatim), without normalization.",
            "Exclude footnote markers/superscripts from the value.",
        ],
        "tie_breakers": [
            "Prefer matches where section/sub-section/line-item cues align.",
            "If multiple, choose the one nearest the hint row.",
        ],
        "REQUIRED_OUTPUT_FORMAT": (
            "Return exactly ONE line of text in this format (no surrounding quotes, no markdown):\n"
            "variable|<value_text>|<detected_cell or NULL>\n"
            "Rules:\n"
            "- Do not print anything else.\n"
            "- If value_text contains the '|' character, escape it as '\\|'.\n"
            "- Remove newlines/tabs from value_text; keep it on one line.\n"
            "- If the cell cannot be inferred, write 'NULL' for detected_cell."
        ),
        "EXAMPLE_ONLY": "Risk Weight|45%|M66"
    }
    return system_text, user_text

################################################################
#2)Add a parser for the pipe line and use it before JSON parsing

def parse_pipe_line(text: Optional[str]) -> Dict[str, Any]:
    if not text:
        raise ValueError("Empty completion text for pipe parsing.")
    # Take the first non-empty line
    line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    # Unwrap accidental code fences or quotes
    if line.startswith("```"):
        line = line.strip("`").strip()
    if line.startswith('"') and line.endswith('"'):
        line = line[1:-1]
    # Split on unescaped pipes
    parts = re.split(r'(?<!\\)\|', line)
    if len(parts) != 3:
        raise ValueError(f"Pipe output did not have 3 fields: {line}")
    variable = parts[0].strip()
    value_text = parts[1].replace("\\|","|").strip()
    detected_cell = parts[2].strip()
    if detected_cell.upper() == "NULL":
        detected_cell = None
    return {"variable": variable, "value_text": value_text, "detected_cell": detected_cell}

#######################################################################
#3) In your call site (right after getting raw_text), swap the parser order
raw_text, raw_resp = call_gemini_openai_style(...)

# First try pipe-delimited parsing (preferred now)
try:
    parsed = parse_pipe_line(raw_text)
except Exception:
    # Fallbacks: try robust JSON (in case the model ignored instructions)
    parsed = coerce_json_from_text(raw_text)

###########################################################################
import re
from typing import Optional, Dict, Any, List, Tuple

CELL_RX = re.compile(r"^[A-Za-z]{1,3}\s*\d{1,5}$")

def _strip_fences_and_quotes(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`").strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    return s

def _unescaped_pipe_positions(s: str) -> List[int]:
    pos = []
    esc = False
    for i, ch in enumerate(s):
        if esc:
            esc = False
            continue
        if ch == '\\':
            esc = True
            continue
        if ch == '|':
            pos.append(i)
    return pos

def _split_by_first_and_last_unescaped_pipe(line: str) -> Optional[Tuple[str, str, str]]:
    line = _strip_fences_and_quotes(line)
    positions = _unescaped_pipe_positions(line)
    if len(positions) < 2:
        return None
    i, j = positions[0], positions[-1]     # allow value segment to contain pipes
    left = line[:i]
    mid  = line[i+1:j]
    right= line[j+1:]
    return left, mid, right

def parse_pipe_line_robust(text: Optional[str]) -> Dict[str, Any]:
    if not text:
        raise ValueError("Empty completion text for pipe parsing.")

    # Consider each non-empty line; prioritize those with >= 2 unescaped pipes
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    candidates = []
    for ln in lines:
        split3 = _split_by_first_and_last_unescaped_pipe(ln)
        if split3:
            candidates.append((ln, split3))

    if not candidates:
        raise ValueError(f"Pipe output did not have 3 fields in any line: {text[:300]}...")

    # Prefer a line where the 3rd field looks like a cell or NULL
    def score(c):
        _ln, (v, val, cell) = c
        cell_t = cell.strip().upper()
        is_cell = bool(CELL_RX.match(cell_t))
        is_null = (cell_t == "NULL")
        return (1 if (is_cell or is_null) else 0)

    ln, (variable, value_text, detected_cell) = sorted(candidates, key=score, reverse=True)[0]

    # Unescape \| in value
    value_text = value_text.replace("\\|", "|").strip()
    variable = variable.strip()

    cell = detected_cell.strip()
    if cell.upper() == "NULL" or not CELL_RX.match(cell):
        cell = None
    else:
        cell = cell.replace(" ", "")  # normalize like "M 66" -> "M66"

    return {"variable": variable, "value_text": value_text, "detected_cell": cell}
############################################################################
raw_text, raw_resp = call_gemini_openai_style(...)

try:
    parsed = parse_pipe_line_robust(raw_text)          # <— primary
except Exception:
    parsed = coerce_json_from_text(raw_text)           # <— fallback if model ignored pipe format

#################################################################################
user_text: Dict[str, Any] = {
    "task": "Extract the variable: '{}'.".format(variable_name),
    "name_pointers": {
        "section_variants": section_variants,
        "subsection_variants": subsection_variants,
        "line_item_variants": line_item_variants,
        "notes": "Variants come from prior reports; use fuzzy matching and layout cues.",
    },
    "cell_hint_and_vicinity": vicinity_desc,
    "layout_prior": (
        "Rows usually correspond to loans and can shift between versions. "
        "Bias vertical scanning near the hint row; then sweep limited columns horizontally."
    ),
    "value_rules": [
        "Return the value exactly as seen (verbatim), without normalization.",
        "Exclude footnote markers/superscripts from the value.",
    ],
    "tie_breakers": [
        "Prefer matches where section/sub-section/line-item cues align.",
        "If multiple, choose the one nearest the hint row.",
    ],

    # 👇 👇 this is the key addition
    "REQUIRED_OUTPUT": (
        "Return exactly ONE line of plain text. "
        "No markdown, no quotes, no code fences. "
        "Format strictly as:\n"
        "<variable>|<value_text>|<detected_cell or NULL>\n\n"
        "Examples:\n"
        "Risk Weight|45%|M66\n"
        "Cash Interest|123,456|NULL\n\n"
        "Rules:\n"
        "- Do not print anything else.\n"
        "- Escape '|' inside value_text as '\\|'.\n"
        "- Keep it all on one line."
    ),
}

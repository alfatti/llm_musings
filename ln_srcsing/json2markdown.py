from typing import Any, Dict, List, Union

def list_to_markdown(items: Union[List[Any], Any]) -> str:
    """
    Turn a list into a markdown bullet list.
    If it's not a list, just stringify it.
    """
    if not isinstance(items, list):
        return str(items)
    return "\n".join(f"- {item}" for item in items)

def dict_to_markdown_prompt(user_text: Dict[str, Any]) -> str:
    """
    Convert your user_text dict into a markdown string.
    Expects keys like:
    - task (str)
    - name_pointers (dict of lists/strings)
    - cell_hint_and_vicinity (str)
    - layout_prior (str)
    - value_rules (list[str])
    - tie_breakers (list[str])
    - REQUIRED_OUTPUT_FORMAT (str)
    - EXAMPLE_ONLY (str)
    """
    parts: List[str] = []

    # Task
    task = user_text.get("task", "")
    parts.append("## Task\n")
    parts.append(task + "\n")

    # Name pointers
    name_pointers = user_text.get("name_pointers", {})
    if isinstance(name_pointers, dict) and name_pointers:
        parts.append("## Name Pointers\n")
        for key, value in name_pointers.items():
            pretty_key = key.replace("_", " ").title()
            parts.append(f"**{pretty_key}**")
            parts.append(list_to_markdown(value) + "\n")

    # Cell hint and vicinity
    vicinity = user_text.get("cell_hint_and_vicinity")
    if vicinity:
        parts.append("## Cell Hint And Vicinity\n")
        parts.append(str(vicinity) + "\n")

    # Layout prior
    layout_prior = user_text.get("layout_prior")
    if layout_prior:
        parts.append("## Layout Prior\n")
        parts.append(str(layout_prior) + "\n")

    # Value rules
    value_rules = user_text.get("value_rules")
    if value_rules:
        parts.append("## Value Rules\n")
        parts.append(list_to_markdown(value_rules) + "\n")

    # Tie breakers
    tie_breakers = user_text.get("tie_breakers")
    if tie_breakers:
        parts.append("## Tie Breakers\n")
        parts.append(list_to_markdown(tie_breakers) + "\n")

    # Required output format (as a code block)
    required_fmt = user_text.get("REQUIRED_OUTPUT_FORMAT")
    if required_fmt:
        parts.append("## Required Output Format\n")
        parts.append("```text")
        parts.append(str(required_fmt))
        parts.append("```")

    # Example line
    example_only = user_text.get("EXAMPLE_ONLY")
    if example_only:
        parts.append("\n## Example\n")
        parts.append(f"`{example_only}`")

    # Join with double newlines for clean markdown
    return "\n".join(parts)


# --- Example usage with your prompt ---

def build_user_text_prompt(
    variable_name: str,
    section_variants,
    subsection_variants,
    line_item_variants,
    vicinity_desc,
) -> str:
    """
    Helper that builds your original dict and converts it to markdown.
    """
    user_text: Dict[str, Any] = {
        "task": f"Extract the variable: '{variable_name}'.",
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
        "EXAMPLE_ONLY": "Risk Weight|45%|M66",
    }

    return dict_to_markdown_prompt(user_text)

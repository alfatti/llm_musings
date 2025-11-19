def dict_to_markdown(data, indent=0):
    """
    Convert any Python dict into a clean, markdown-friendly text block.
    - Nested dicts are indented
    - Lists become bullet points
    - Everything else becomes plain text
    """

    lines = []
    pad = "  " * indent

    if isinstance(data, dict):
        for key, value in data.items():
            lines.append(f"{pad}- **{key}**:")
            lines.append(dict_to_markdown(value, indent + 1))

    elif isinstance(data, list):
        for item in data:
            lines.append(f"{pad}- {dict_to_markdown(item, indent + 1).strip()}")

    else:
        lines.append(f"{pad}{data}")

    return "\n".join(lines)

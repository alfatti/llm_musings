def json_prompt_to_markdown(prompt_json: dict) -> str:
    """
    Convert a structured JSON prompt (with system/user/assistant fields)
    into a readable Markdown format.

    Example input:
    {
        "system_prompt": "You are an AI that extracts rent roll data.",
        "user_prompt": "Extract all unit rows from this text: {{page_text}}"
    }

    Example output:
    ### System
    You are an AI that extracts rent roll data.

    ### User
    Extract all unit rows from this text: {{page_text}}
    """
    import json

    # Define mapping from JSON keys to Markdown section headers
    role_map = {
        "system_prompt": "System",
        "user_prompt": "User",
        "assistant_prompt": "Assistant",
        "developer_prompt": "Developer"
    }

    md_blocks = []
    for key, value in prompt_json.items():
        if value is None or str(value).strip() == "":
            continue  # skip empty fields

        # Use known mapping or fallback to a title-cased version of the key
        header = role_map.get(key, key.replace("_", " ").title())
        block = f"### {header}\n{value.strip()}\n"
        md_blocks.append(block)

    markdown_text = "\n".join(md_blocks).strip()
    return markdown_text
def json_prompt_to_markdown(prompt_json: dict) -> str:
    """
    Convert a structured JSON prompt (with system/user/assistant fields)
    into a readable Markdown format.

    Example input:
    {
        "system_prompt": "You are an AI that extracts rent roll data.",
        "user_prompt": "Extract all unit rows from this text: {{page_text}}"
    }

    Example output:
    ### System
    You are an AI that extracts rent roll data.

    ### User
    Extract all unit rows from this text: {{page_text}}
    """
    import json

    # Define mapping from JSON keys to Markdown section headers
    role_map = {
        "system_prompt": "System",
        "user_prompt": "User",
        "assistant_prompt": "Assistant",
        "developer_prompt": "Developer"
    }

    md_blocks = []
    for key, value in prompt_json.items():
        if value is None or str(value).strip() == "":
            continue  # skip empty fields

        # Use known mapping or fallback to a title-cased version of the key
        header = role_map.get(key, key.replace("_", " ").title())
        block = f"### {header}\n{value.strip()}\n"
        md_blocks.append(block)

    markdown_text = "\n".join(md_blocks).strip()
    return markdown_text

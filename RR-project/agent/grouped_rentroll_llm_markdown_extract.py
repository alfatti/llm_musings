
import pandas as pd
import json
from typing import List, Dict

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser


def load_unit_blocks_as_markdown(path: str) -> List[str]:
    df = pd.read_excel(path, header=None)

    # Step 1: Extract first non-empty row as header
    for i, row in df.iterrows():
        if not row.isnull().all():
            header = row.tolist()
            break
    else:
        raise ValueError("No non-empty header row found")

    # Step 2: Drop the header row and process blocks
    df = df.iloc[i + 1:].reset_index(drop=True)

    blocks = []
    current_block = []

    for _, row in df.iterrows():
        if row.isnull().all():
            if current_block:
                df_block = pd.DataFrame(current_block).ffill(axis=0)
                df_block.columns = header[:df_block.shape[1]]
                markdown = df_block.to_markdown(index=False)
                blocks.append(markdown)
                current_block = []
        else:
            current_block.append(row)

    if current_block:
        df_block = pd.DataFrame(current_block).ffill(axis=0)
        df_block.columns = header[:df_block.shape[1]]
        markdown = df_block.to_markdown(index=False)
        blocks.append(markdown)

    return blocks


def build_batch_prompt(markdown_tables: List[str]) -> str:
    joined_tables = "\n\n---\n\n".join(f"Table {i+1}:\n\n{table}" for i, table in enumerate(markdown_tables))
    prompt = (
        "You are a rent-roll extractor.\n\n"
        "Each table below represents a single unit, with multiple charge types and shared columns like tenant info.\n\n"
        "For each table, extract a JSON object with these fields:\n"
        "- unit (from Unit column)\n"
        "- tenant_name\n"
        "- lease_start\n"
        "- lease_end\n"
        "- gla\n"
        "- base_rent: amount corresponding to the row where charge code includes 'base rent' (case-insensitive)\n\n"
        "Return a JSON list of unit records. Example:\n"
        "[{\"unit\": \"101\", \"tenant_name\": \"John Smith\", \"lease_start\": \"2024-06-01\", "
        "\"lease_end\": \"2025-05-31\", \"gla\": 496, \"base_rent\": 2350.0}, ...]\n\n"
        f"{joined_tables}"
    )
    return prompt


def extract_llm_output(prompt: str) -> List[dict]:
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
    chain = ChatPromptTemplate.from_template("{input}") | llm | StrOutputParser()
    response = chain.invoke({"input": prompt})

    try:
        parsed = json.loads(response)
        if not isinstance(parsed, list):
            raise ValueError("Expected a JSON list.")
        return parsed
    except Exception as e:
        print("Failed to parse LLM response:")
        print(response)
        raise e


def chunk_list(data: List, chunk_size: int) -> List[List]:
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


def main():
    excel_path = "rent_roll.xlsx"
    all_markdown_blocks = load_unit_blocks_as_markdown(excel_path)
    markdown_batches = chunk_list(all_markdown_blocks, chunk_size=10)

    all_units = []

    for i, batch in enumerate(markdown_batches):
        print(f"Processing batch {i + 1} of {len(markdown_batches)}...")
        prompt = build_batch_prompt(batch)
        result = extract_llm_output(prompt)
        all_units.extend(result)

    with open("extracted_units.json", "w") as f:
        json.dump(all_units, f, indent=2)

    print("✅ Extraction complete. Output saved to extracted_units.json")


if __name__ == "__main__":
    main()

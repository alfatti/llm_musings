
import pandas as pd
import json
from typing import List, Dict

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser


def load_grouped_units_with_charges(path: str) -> List[Dict]:
    df = pd.read_excel(path, header=None)

    groups = []
    current_group = []

    for _, row in df.iterrows():
        if row.isnull().all():
            if current_group:
                groups.append(pd.DataFrame(current_group))
                current_group = []
        else:
            current_group.append(row)

    if current_group:
        groups.append(pd.DataFrame(current_group))

    parsed_units = []

    for group in groups:
        group_ffill = group.ffill(axis=0)

        group_ffill.columns = ["unit", "charge_code", "amount", "lease_start", "lease_end", "gla"]

        charges = []
        for _, row in group_ffill.iterrows():
            charge_code = str(row["charge_code"]).strip()
            amount = row["amount"]
            if pd.isna(charge_code) or pd.isna(amount):
                continue
            charges.append({
                "charge_code": charge_code,
                "amount": float(amount)
            })

        shared_row = group_ffill.iloc[0]
        unit_info = {
            "unit": str(shared_row["unit"]).strip(),
            "lease_start": str(shared_row["lease_start"]),
            "lease_end": str(shared_row["lease_end"]),
            "gla": shared_row["gla"],
            "charges": charges
        }
        parsed_units.append(unit_info)

    return parsed_units


def build_prompt(units: List[Dict]) -> str:
    data_json = json.dumps(units, indent=2)
    prompt = (
        "Below are rent-roll records for several units in JSON format. Each record includes:\n"
        "- unit-level details (unit, lease_start, lease_end, gla)\n"
        "- a list of charges (with charge_code and amount)\n\n"
        "Your task:\n"
        "For each unit, extract:\n"
        "- unit\n"
        "- lease_start\n"
        "- lease_end\n"
        "- gla\n"
        "- base_rent: the amount associated with the charge_code containing \"base rent\" (case-insensitive)\n\n"
        "Return a JSON list of such simplified unit records.\n\n"
        f"```json\n{data_json}\n```"
    )
    return prompt


def extract_chunk(prompt: str) -> List[dict]:
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


def split_list(data: List, chunk_size: int) -> List[List]:
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


def main():
    excel_path = "rent_roll.xlsx"
    units = load_grouped_units_with_charges(excel_path)

    unit_chunks = split_list(units, chunk_size=25)
    all_results = []

    for i, chunk in enumerate(unit_chunks):
        print(f"Processing chunk {i + 1} of {len(unit_chunks)}...")
        prompt = build_prompt(chunk)
        result = extract_chunk(prompt)
        all_results.extend(result)

    with open("extracted_rentroll.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("Extraction complete. Output saved to extracted_rentroll.json.")


if __name__ == "__main__":
    main()

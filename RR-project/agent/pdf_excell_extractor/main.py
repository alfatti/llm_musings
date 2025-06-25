### file: main.py
"""Command‑line entry point: route file → chunks → LangGraph loop → DataFrame."""

import argparse, json
from pathlib import Path

import pandas as pd

from file_router import get_file_type
from parsers import PdfPreparser, ExcelPreparser
from graph_builder import build_workflow


def _load_chunks(path: str, file_type: str):
    if file_type == "pdf":
        return PdfPreparser(path).get_chunks()
    return ExcelPreparser(path).get_chunks()


def main():
    parser = argparse.ArgumentParser(
        description="Rent‑roll extractor for PDF & Excel with Gemini"
    )
    parser.add_argument("file", help="Path to PDF or Excel rent roll")
    parser.add_argument(
        "-o",
        "--out",
        default="rentroll.parquet",
        help="Output file (.parquet or .csv)",
    )
    args = parser.parse_args()

    file_type = get_file_type(args.file)
    chunks = _load_chunks(args.file, file_type)
    if not chunks:
        raise RuntimeError("No text chunks extracted. Check file layout.")

    workflow = build_workflow()
    rows = []

    for i, chunk in enumerate(chunks, 1):
        state = {"chunk_text": chunk, "file_type": file_type}
        final = workflow.invoke(state)
        extracted_json = json.loads(final["extracted"])
        for row in extracted_json:
            row["source_chunk"] = i
            rows.append(row)

    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    if out_path.suffix == ".csv":
        df.to_csv(out_path, index=False)
    else:
        df.to_parquet(out_path, index=False)

    print(f"✅ Saved {len(df)} rows → {out_path}")


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────────
#  End of package.  Required deps:
#  pip install langchain langgraph google-generativeai pypdf pandas openpyxl jsonschema
# ─────────────────────────────────────────────────────────────────────────────

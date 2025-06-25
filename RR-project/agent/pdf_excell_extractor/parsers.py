### file: parsers.py
from __future__ import annotations
from typing import List
from pathlib import Path

import pandas as pd
from pypdf import PdfReader


class PdfPreparser:
    """Extract raw text *per page* from a PDF."""

    def __init__(self, path: str):
        self.reader = PdfReader(path)

    def get_chunks(self) -> List[str]:
        return [page.extract_text() for page in self.reader.pages]


class ExcelPreparser:
    """Chunk an Excel rent‑roll by blank‑line separators (common layout)."""

    def __init__(self, path: str):
        # engine="openpyxl" keeps xls/xlsx parity
        self.df = pd.read_excel(path, sheet_name=0, engine="openpyxl")

    def _chunk_bounds(self):
        blank = self.df.index[self.df.isna().all(axis=1)].tolist()
        blank = [-1] + blank + [len(self.df)]
        for lo, hi in zip(blank[:-1], blank[1:]):
            yield lo + 1, hi

    def get_chunks(self) -> List[str]:
        chunks: List[str] = []
        for lo, hi in self._chunk_bounds():
            chunk_df = self.df.iloc[lo:hi].dropna(how="all")
            if chunk_df.empty:
                continue
            # Tab‑delimited row text keeps column order visible to the LLM
            chunk_text = "\n".join(
                "\t".join(map(str, row)) for row in chunk_df.astype(str).values
            )
            chunks.append(chunk_text)
        return chunks

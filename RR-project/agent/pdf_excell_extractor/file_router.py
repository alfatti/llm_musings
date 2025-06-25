### file: file_router.py
from pathlib import Path


def get_file_type(file_path: str) -> str:
    """Return 'pdf' or 'excel' based on file extension."""
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    elif suffix in {".xls", ".xlsx"}:
        return "excel"
    raise ValueError(f"Unsupported file type: {suffix}")

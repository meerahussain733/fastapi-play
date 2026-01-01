from pathlib import Path
from typing import List, Dict

def load_txt_documents(docs_dir: str = "docs_rag") -> List[Dict[str, str]]:
    """
    Loads all .txt files from docs_dir and returns a list of dicts:
    [{"source": "filename.txt", "text": "..."}]
    """
    base = Path(docs_dir)
    if not base.exists():
        raise FileNotFoundError(f"Docs folder not found: {base.resolve()}")

    docs: List[Dict[str, str]] = []
    for p in sorted(base.glob("*.txt")):
        text = p.read_text(encoding="utf-8", errors="ignore")
        docs.append({"source": p.name, "text": text})

    return docs

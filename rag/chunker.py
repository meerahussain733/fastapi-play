from typing import List, Dict

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks.

    - chunk_size: max characters per chunk
    - overlap: how many characters to repeat between chunks (keeps context)
    """
    text = (text or "").strip()
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == n:
            break

        # move start forward but keep overlap
        start = max(end - overlap, start + 1)

    return chunks


def chunk_documents(
    docs: List[Dict[str, str]],
    chunk_size: int = 800,
    overlap: int = 100
) -> List[Dict[str, str]]:
    """
    Input docs:  [{"source": "...", "text": "..."}]
    Output chunks: [{"source": "...", "chunk_id": 0, "text": "..."}, ...]
    """
    all_chunks: List[Dict[str, str]] = []

    for d in docs:
        source = d["source"]
        pieces = chunk_text(d["text"], chunk_size=chunk_size, overlap=overlap)

        for i, piece in enumerate(pieces):
            all_chunks.append({
                "source": source,
                "chunk_id": i,
                "text": piece,
            })

    return all_chunks

from typing import List, Dict, Tuple
import numpy as np

def top_k_similar(
    query_vec: np.ndarray,
    doc_vecs: np.ndarray,
    k: int = 3
) -> List[Tuple[int, float]]:
    """
    Returns list of (index, score) for top-k most similar vectors.
    Assumes vectors are L2-normalized so dot product == cosine similarity.
    """
    if doc_vecs.size == 0:
        return []

    # scores shape: (N,)
    scores = doc_vecs @ query_vec

    k = min(k, scores.shape[0])
    top_idx = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i])) for i in top_idx]


def retrieve_top_k(
    query: str,
    chunks: List[Dict[str, str]],
    chunk_vecs: np.ndarray,
    embed_fn,
    k: int = 3
) -> List[Dict[str, object]]:
    """
    query -> embed -> top-k chunk matches
    embed_fn should accept List[str] and return np.ndarray.
    """
    q_vec = embed_fn([query])[0]  # shape (D,)
    ranked = top_k_similar(q_vec, chunk_vecs, k=k)

    results = []
    for idx, score in ranked:
        c = chunks[idx]
        results.append({
            "source": c["source"],
            "chunk_id": c["chunk_id"],
            "score": score,
            "text": c["text"],
        })
    return results

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# load once
_model = SentenceTransformer(MODEL_NAME)

def embed_texts(texts: List[str]) -> np.ndarray:
    """
    texts -> vectors (N, D). D is usually 384 for this model.
    normalize_embeddings=True makes cosine similarity easy later.
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    vectors = _model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)

    return vectors

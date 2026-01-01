from typing import List, Dict
import numpy as np

from rag.loader import load_txt_documents
from rag.chunker import chunk_documents
from rag.embedder import embed_texts


class RAGStore:
    def __init__(self, docs_dir: str = "docs", chunk_size: int = 800, overlap: int = 100):
        self.docs_dir = docs_dir
        self.chunk_size = chunk_size
        self.overlap = overlap

        self.chunks: List[Dict[str, str]] = []
        self.vectors: np.ndarray = np.zeros((0, 0), dtype=np.float32)

    def build(self):
        docs = load_txt_documents(self.docs_dir)
        self.chunks = chunk_documents(docs, chunk_size=self.chunk_size, overlap=self.overlap)
        self.vectors = embed_texts([c["text"] for c in self.chunks])

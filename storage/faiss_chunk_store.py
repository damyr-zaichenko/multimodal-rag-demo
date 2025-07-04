import faiss
import numpy as np
import pickle
from typing import List, Tuple


class FaissChunkStore:
    def __init__(self, dim: int, index_path: str = "data/faiss_index.bin", metadata_path: str = "data/metadata.pkl"):
        self.dim = dim
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = faiss.IndexFlatL2(dim)
        self.metadata: List[str] = []

        # Load if exists
        self._load()

    def add(self, embedding: np.ndarray, metadata: dict):
        if metadata in self.metadata:
            print(f"Chunk already exists. Skipping.")
            return
        self.index.add(np.array([embedding], dtype=np.float32))
        self.metadata.append(metadata)

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        D, I = self.index.search(np.array([query_vector], dtype=np.float32), k)
        return [(self.metadata[i], float(D[0][j])) for j, i in enumerate(I[0]) if i < len(self.metadata)]

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def _load(self):
        try:
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "rb") as f:
                self.metadata = pickle.load(f)
        except Exception:
            print("No existing FAISS index found. Starting fresh.")

    def get_all_urls(self) -> list[str]:
        return [meta['article_url'] for meta in self.metadata]
    
    def get_metadata(self):
        return self.metadata
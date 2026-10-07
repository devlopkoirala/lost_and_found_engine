import os
import json
import numpy as np
import faiss
from typing import List, Dict, Union, Tuple


class VectorSearchEngine:
    def __init__(self, dimension: int = 512, index_type: str = "FlatIP"):
        """
        Initializes the FAISS search engine.
        :param dimension: Dimensionality of embeddings (512 for CLIP ViT-B/32).
        :param index_type: 'FlatIP' for Cosine Similarity or 'FlatL2' for Euclidean.
        """
        self.dimension = dimension
        self.index_type = index_type
        self.id_to_sku: Dict[int, str] = {}
        self.index = self._build_base_index()

    def _build_base_index(self) -> faiss.Index:
        if self.index_type == "FlatIP":
            base_index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "FlatL2":
            base_index = faiss.IndexFlatL2(self.dimension)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        # IndexIDMap2 permits explicit tracking and deletion of custom 64-bit IDs
        return faiss.IndexIDMap2(base_index)

    def add_vectors(self, embeddings: np.ndarray, sku_ids: List[str]) -> None:
        """
        Normalizes embeddings, maps SKU strings to numeric IDs, and adds them to FAISS.
        """
        if len(embeddings) != len(sku_ids):
            raise ValueError("Number of embeddings does not match number of SKU IDs.")

        # FAISS requires contiguous float32 memory layout
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        if self.index_type == "FlatIP":
            faiss.normalize_L2(embeddings)

        start_id = len(self.id_to_sku)
        numeric_ids = np.arange(start_id, start_id + len(sku_ids), dtype=np.int64)

        for num_id, sku in zip(numeric_ids, sku_ids):
            self.id_to_sku[int(num_id)] = sku

        self.index.add_with_ids(embeddings, numeric_ids)

    def search(
        self, query_vector: np.ndarray, k: int = 5
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Executes a top-K similarity search for a given query vector.
        """
        query_vector = np.ascontiguousarray(query_vector, dtype=np.float32)

        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)

        if self.index_type == "FlatIP":
            faiss.normalize_L2(query_vector)

        distances, indices = self.index.search(query_vector, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 if requested K exceeds index size
                continue
            results.append({
                "sku_id": self.id_to_sku[int(idx)],
                "score": float(dist)
            })

        return results

    def save_index(self, index_path: str, mapping_path: str) -> None:
        """Persists the FAISS binary index and SKU mapping dictionary to disk."""
        faiss.write_index(self.index, index_path)
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(self.id_to_sku, f, indent=2)

    def load_index(self, index_path: str, mapping_path: str) -> None:
        """Loads a pre-built FAISS binary index and SKU mapping dictionary from disk."""
        if not os.path.exists(index_path) or not os.path.exists(mapping_path):
            raise FileNotFoundError("Index or mapping file path does not exist.")

        self.index = faiss.read_index(index_path)
        with open(mapping_path, "r", encoding="utf-8") as f:
            raw_map = json.load(f)
            self.id_to_sku = {int(k): v for k, v in raw_map.items()}


if __name__ == "__main__":
    # Smoke Test / Demonstration
    N, D = 1000, 512
    fake_embeddings = np.random.randn(N, D).astype(np.float32)
    fake_skus = [f"SKU_{i:05d}" for i in range(N)]

    engine = VectorSearchEngine(dimension=D, index_type="FlatIP")
    engine.add_vectors(fake_embeddings, fake_skus)

    # Simulate query vector
    query = fake_embeddings[0]
    matches = engine.search(query, k=3)

    print("Top Search Results:")
    for rank, match in enumerate(matches, start=1):
        print(f"{rank}. SKU: {match['sku_id']} | Score: {match['score']:.4f}")
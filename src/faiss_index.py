"""
FAISS Vector Store Module

Provides FAISSIndex class with:
- add_chunks
- search (with optional product_id filter)
- save/load
- stats
"""

from typing import List, Dict, Any, Optional
import numpy as np
import faiss
import json
import os


class FAISSIndex:
    def __init__(self, dimension: int, index_type: str = "flat"):
        self.dimension = dimension
        self.index_type = index_type
        self.metadata: List[Dict[str, Any]] = []

        if index_type == "flat":
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
            self.index.nprobe = 10
        else:
            raise ValueError("index_type must be 'flat' or 'ivf'")

    # ---------------------------------------------------------
    # Add vectors + parallel metadata
    # ---------------------------------------------------------
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        if not chunks:
            return

        vectors = []
        for i, chunk in enumerate(chunks):
            emb = chunk.get("embedding")
            if emb is None:
                raise ValueError(f"Chunk {i} missing 'embedding'")
            if not isinstance(emb, np.ndarray):
                emb = np.array(emb, dtype=np.float32)
            vectors.append(emb.astype(np.float32))

            # store everything except the embedding as metadata
            meta = {k: v for k, v in chunk.items() if k != "embedding"}
            self.metadata.append(meta)

        matrix = np.vstack(vectors).astype(np.float32)
        self.index.add(matrix)

    # ---------------------------------------------------------
    # Search with optional product_id filter
    # ---------------------------------------------------------
    def search(
        self,
        query_vector,
        k: int = 5,
        filter_product_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search the FAISS index.

        Args:
            query_vector: numpy array or list
            k: number of results to return AFTER filtering
            filter_product_id: if set, only return chunks where metadata['product_id']
                               matches this value.

        Returns:
            List of metadata dicts (each with optional 'distance' added).
        """
        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector, dtype=np.float32)
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        # search for more than k to give the filter room
        search_k = max(k * 3, k)
        distances, indices = self.index.search(query_vector, search_k)

        results: List[Dict[str, Any]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if not (0 <= idx < len(self.metadata)):
                continue

            meta = self.metadata[idx]

            if filter_product_id is not None:
                if meta.get("product_id") != filter_product_id:
                    continue

            # copy to avoid mutating stored metadata
            out = dict(meta)
            out["distance"] = float(dist)
            results.append(out)

            if len(results) >= k:
                break

        return results

    # ---------------------------------------------------------
    # Save / Load
    # ---------------------------------------------------------
    def save(self, index_path: str, metadata_path: str) -> None:
        os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

    def load(self, index_path: str, metadata_path: str) -> None:
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError("Index or metadata file not found")
        self.index = faiss.read_index(index_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    # ---------------------------------------------------------
    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": self.index_type,
        }

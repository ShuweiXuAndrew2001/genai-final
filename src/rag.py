"""
RAG (Retrieval-Augmented Generation) Module

RAGPipeline:
- embed query with Mistral
- search FAISSIndex with product_id filter
- call GroqLLM with contextual prompt
"""

from typing import List, Dict, Any
from .embeddings import MistralEmbedder
from .faiss_index import FAISSIndex
from .groq_llm import GroqLLM


class RAGPipeline:
    def __init__(self, faiss_index: FAISSIndex, embedder: MistralEmbedder, llm: GroqLLM):
        self.index = faiss_index
        self.embedder = embedder
        self.llm = llm

    def retrieve(
        self,
        query: str,
        product_id: str,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k chunks for a given query, restricted to one product.
        """
        query_vec = self.embedder.embed_text(query)
        # search with product_id filter
        results = self.index.search(
            query_vec,
            k=k,
            filter_product_id=product_id,
        )
        return results

    def refine(self, query: str, product_id: str, k: int = 5) -> str:
        """
        Run retrieval + LLM refinement for a single product.
        """
        retrieved = self.retrieve(query, product_id=product_id, k=k)
        return self.llm.refine_with_rag(query, retrieved)

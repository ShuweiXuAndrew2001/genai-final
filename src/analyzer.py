"""
Product Analyzer Module

Orchestrates:
- Chunking
- Embedding
- Indexing
- RAG refinement (Groq + FAISS)
- Visual attribute extraction
- Image prompt generation
"""

from typing import List, Dict, Any, Optional
import json
import os

from .chunker import TextChunker
from .embeddings import MistralEmbedder
from .faiss_index import FAISSIndex
from .rag import RAGPipeline
from .groq_llm import GroqLLM


class ProductAnalyzer:
    """
    Main product analysis pipeline.
    """

    def __init__(
        self,
        mistral_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        min_tokens: int = 800,
        max_tokens: int = 1200,
        index_type: str = "flat",
    ):
        self.chunker = TextChunker(min_tokens=min_tokens, max_tokens=max_tokens)
        self.embedder = MistralEmbedder(api_key=mistral_api_key)

        dim = self.embedder.get_embedding_dimension()
        self.index = FAISSIndex(dimension=dim, index_type=index_type)

        self.llm = GroqLLM(api_key=groq_api_key)

        self.rag = RAGPipeline(
            faiss_index=self.index,
            embedder=self.embedder,
            llm=self.llm,
        )

        self._is_indexed = False

    # ---------------------------------------------------------
    # Build index from all products
    # ---------------------------------------------------------
    def process_products(self, products: List[Dict[str, Any]]) -> None:
        all_chunks = []

        for product in products:
            description = product.get("description", "").strip()
            desc_tokens = (
                self.chunker.count_tokens(description) if description else 0
            )

            chunks = self.chunker.chunk_product_data(product)

            # Count description chunks
            desc_chunk_count = 0
            if description:
                accumulated_tokens = 0
                for chunk in chunks:
                    t = self.chunker.count_tokens(chunk.get("text", ""))
                    accumulated_tokens += t
                    desc_chunk_count += 1
                    if accumulated_tokens >= desc_tokens * 0.9:
                        break

            for idx, chunk in enumerate(chunks):
                chunk["product_id"] = product.get("id", "unknown")
                chunk["product_name"] = product.get("name", "unknown")
                chunk["category"] = product.get("category", "unknown")
                chunk["source"] = "description" if idx < desc_chunk_count else "review"

            all_chunks.extend(chunks)

        print(f"🔹 Generating embeddings for {len(all_chunks)} chunks...")
        chunks_with_embeddings = self.embedder.embed_chunks(all_chunks)

        print(f"🔹 Indexing chunks into FAISS...")
        self.index.add_chunks(chunks_with_embeddings)
        self._is_indexed = True
        print(f"✅ Indexed {self.index.index.ntotal} vectors")

    # ---------------------------------------------------------
    # Analyze a single product
    # ---------------------------------------------------------
    def analyze_product(
        self,
        product: Dict[str, Any],
        analysis_queries: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self._is_indexed:
            raise RuntimeError("Index not built. Call process_products() first.")

        product_id = product.get("id", "unknown")

        results: Dict[str, Any] = {
            "product_id": product_id,
            "product_name": product.get("name", "Unknown"),
            "category": product.get("category", "Unknown"),
        }

        # Step 1: Summarization
        print(f"📝 Step 1: Summarizing product {product_id}...")
        product_text = self._get_product_text(product)
        summary = self.llm.summarize(product_text)
        results["summary"] = summary

        # Step 2: Structured JSON extraction
        print(f"📊 Step 2: JSON extraction for {product_id}...")
        try:
            structured_data = self.llm.extract_json(product_text)
        except Exception as e:
            print(f"Warning: JSON extraction failed for {product_id}: {e}")
            structured_data = {}

        results["structured_data"] = structured_data

        if output_dir:
            extracted_path = os.path.join(
                output_dir, "extracted", f"{product_id}_extracted.json"
            )
            os.makedirs(os.path.dirname(extracted_path), exist_ok=True)
            with open(extracted_path, "w", encoding="utf-8") as f:
                json.dump(structured_data, f, indent=2, ensure_ascii=False)

        # Step 3: RAG refinement with analysis queries (NOW PRODUCT-SPECIFIC)
        print(f"🔍 Step 3: RAG analysis for {product_id}...")
        if analysis_queries is None:
            analysis_queries = [
                "What are the key visual and aesthetic characteristics of this product?",
                "What do customers like most about this product?",
                "What are the main use cases and contexts?",
                "What materials, colors, and design elements are mentioned?",
            ]

        rag_results: Dict[str, str] = {}
        for q in analysis_queries:
            answer = self.rag.refine(q, product_id=product_id, k=5)
            rag_results[q] = answer

        results["rag_analysis"] = rag_results

        # Step 4: Visual attributes extraction
        print(f"🎨 Step 4: Visual attributes for {product_id}...")
        analysis_summary = summary + "\n\n" + "\n\n".join(
            [f"Q: {q}\nA: {a}" for q, a in rag_results.items()]
        )

        try:
            visual_attributes = self.llm.generate_visual_attributes(
                product_data=product,
                analysis_summary=analysis_summary,
            )
        except Exception as e:
            print(f"Warning: Visual attribute extraction failed for {product_id}: {e}")
            visual_attributes = {}

        results["visual_attributes"] = visual_attributes

        # Step 5: Image prompt generation
        print(f"🖼  Step 5: Image prompt for {product_id}...")
        if visual_attributes:
            try:
                img_prompt = self.llm.generate_image_prompt(
                    product_data=product,
                    visual_attributes=visual_attributes,
                )
            except Exception as e:
                print(f"Warning: Image prompt generation failed for {product_id}: {e}")
                img_prompt = ""
        else:
            img_prompt = ""

        results["image_prompt"] = img_prompt

        if output_dir:
            prompt_path = os.path.join(
                output_dir, "prompts", f"{product_id}_prompt.txt"
            )
            os.makedirs(os.path.dirname(prompt_path), exist_ok=True)
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write(img_prompt)

        return results

    # ---------------------------------------------------------
    # Analyze all products
    # ---------------------------------------------------------
    def analyze_all_products(
        self,
        products: List[Dict[str, Any]],
        analysis_queries: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        all_results = []
        for i, product in enumerate(products, 1):
            print(f"\n[{i}/{len(products)}] {product.get('name', 'Unknown')}")
            result = self.analyze_product(product, analysis_queries, output_dir)
            all_results.append(result)
        return all_results

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------
    def _get_product_text(self, product: Dict[str, Any]) -> str:
        parts = []
        if product.get("name"):
            parts.append(f"Product Name: {product['name']}")
        if product.get("description"):
            parts.append(f"Description: {product['description']}")
        reviews = product.get("reviews", [])
        if reviews:
            parts.append("Customer Reviews:")
            for i, review in enumerate(reviews, 1):
                parts.append(f"Review {i}: {review}")
        return "\n\n".join(parts)

    def save_index(self, index_path: str, metadata_path: str) -> None:
        self.index.save(index_path, metadata_path)

    def load_index(self, index_path: str, metadata_path: str) -> None:
        self.index.load(index_path, metadata_path)
        self._is_indexed = True

    def get_index_stats(self) -> Dict[str, Any]:
        return self.index.get_stats()

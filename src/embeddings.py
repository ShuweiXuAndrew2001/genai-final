"""
Embeddings Module — FINAL STABLE VERSION

Compatible with your installed Mistral SDK.

Key points:
- Uses `Mistral` client (not MistralClient)
- Correct call: `client.embeddings.create(model=..., inputs=[...])`
- Batches all texts in a single call
- Handles 429 / service_tier_capacity_exceeded with:
    - Retry + small backoff
    - Fallback to alternate embedding models
"""

import os
import time
from typing import List, Optional

import numpy as np
from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()


class MistralEmbedder:
    """
    Wrapper around Mistral embeddings with:
    - batching
    - retry
    - model fallback
    """

    # Ordered by preference. All of these exist in your model list.
    CANDIDATE_MODELS = [
        "mistral-embed",
        "mistral-embed-2312",
        "codestral-embed",
        "codestral-embed-2505",
    ]

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("Missing MISTRAL_API_KEY in environment or .env")

        self.client = Mistral(api_key=self.api_key)

        # If a specific model is forced, use that; otherwise start with the first candidate
        self.model = model or self.CANDIDATE_MODELS[0]

    # ------------------------------------------------------------------
    # Low-level: embed a batch of texts with a specific model
    # ------------------------------------------------------------------
    def _embed_batch_with_model(
        self,
        texts: List[str],
        model: str,
        max_retries: int = 3,
        base_sleep: float = 1.0,
    ) -> List[np.ndarray]:
        """
        Call Mistral embeddings.create with retry for a specific model.
        """
        last_err = None

        for attempt in range(max_retries):
            try:
                # Mistral SDK: embeddings.create(model=..., inputs=[...])
                response = self.client.embeddings.create(
                    model=model,
                    inputs=texts,
                )
                vectors: List[np.ndarray] = []
                for item in response.data:
                    vectors.append(np.array(item.embedding, dtype=np.float32))
                return vectors

            except Exception as e:
                last_err = e
                msg = str(e).lower()
                # Check for capacity / 429 errors
                if "service_tier_capacity_exceeded" in msg or "429" in msg:
                    sleep_time = base_sleep * (attempt + 1)
                    print(
                        f"⚠️ Capacity/429 on model '{model}', attempt {attempt+1}/{max_retries}. "
                        f"Sleeping {sleep_time:.1f}s..."
                    )
                    time.sleep(sleep_time)
                    continue
                else:
                    # Not a capacity error → re-raise immediately
                    raise

        # If we exhausted retries for this model
        raise last_err or RuntimeError(
            f"Failed to embed with model {model} after {max_retries} retries"
        )

    # ------------------------------------------------------------------
    # Public: embed a single text
    # ------------------------------------------------------------------
    def embed_text(self, text: str) -> np.ndarray:
        """
        Convenience wrapper over embed_batch for a single string.
        """
        if not text or not text.strip():
            # Empty text → zero vector
            return np.zeros((self.get_embedding_dimension(),), dtype=np.float32)

        vectors = self.embed_batch([text])
        return vectors[0]

    # ------------------------------------------------------------------
    # Public: embed a list of texts with model fallback
    # ------------------------------------------------------------------
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Embed a list of texts.

        Strategy:
        - Try the current model (self.model)
        - If we get a capacity error, fall back to the next candidate model
        - If all models fail with capacity errors, raise a clear error
        """
        if not texts:
            return []

        # Normalize texts (avoid None)
        texts = [t if t is not None else "" for t in texts]

        # Try primary + fallback models
        last_error = None
        for model in self.CANDIDATE_MODELS:
            print(f"🧠 Trying embedding model: {model}")
            try:
                vectors = self._embed_batch_with_model(texts, model=model)
                # If success, remember working model for future calls
                self.model = model
                print(f"✅ Using embedding model: {model}")
                return vectors
            except Exception as e:
                msg = str(e).lower()
                last_error = e
                if "service_tier_capacity_exceeded" in msg or "429" in msg:
                    print(f"⚠️ Model '{model}' at capacity, trying next candidate...")
                    continue
                else:
                    # Hard failure (invalid model, auth, etc.) → don't try further
                    raise

        # If we reach here, all models failed due to capacity / 429
        raise RuntimeError(
            f"All candidate embedding models are at capacity or failed with 429. "
            f"Last error: {last_error}"
        )

    # ------------------------------------------------------------------
    # Public: embed chunks (list of dicts with 'text')
    # ------------------------------------------------------------------
    def embed_chunks(self, chunks: List[dict]) -> List[dict]:
        if not chunks:
            return []

        texts = [c.get("text", "") for c in chunks]
        vectors = self.embed_batch(texts)

        for chunk, vec in zip(chunks, vectors):
            chunk["embedding"] = vec

        return chunks

    # ------------------------------------------------------------------
    def get_embedding_dimension(self) -> int:
        """
        Mistral embed models are 1024-d.
        """
        return 1024

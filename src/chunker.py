"""
Text Chunking Module
Deterministic, token-aware chunking for product descriptions + reviews.
"""

from typing import List, Dict, Any
import tiktoken


class TextChunker:
    """
    Deterministic text chunker using tiktoken for token counting.

    Never splits mid-sentence.
    Targets 800–1200 tokens per chunk.
    Automatically merges small reviews.
    """

    def __init__(
        self,
        min_tokens: int = 800,
        max_tokens: int = 1200,
        encoding_name: str = "cl100k_base"
    ):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            raise ValueError(f"Failed to load tiktoken encoding: {e}")

    # ---------------------------------------------
    # Token counter
    # ---------------------------------------------
    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self.encoding.encode(text))

    # ---------------------------------------------
    # High-level: chunk whole product
    # ---------------------------------------------
    def chunk_product_data(self, product: Dict[str, Any]) -> List[Dict[str, Any]]:
        chunks = []
        chunk_id = 0

        # 1. Description
        description = product.get("description", "").strip()
        if description:
            desc_chunks = self._chunk_text(description, start_chunk_id=chunk_id)
            for c in desc_chunks:
                c["source"] = "description"
            chunks.extend(desc_chunks)
            chunk_id += len(desc_chunks)

        # 2. Reviews (merged)
        reviews = product.get("reviews", [])
        if reviews:
            review_chunks = self._chunk_reviews(reviews, start_chunk_id=chunk_id)
            for c in review_chunks:
                c["source"] = "review"
            chunks.extend(review_chunks)

        return chunks

    # ---------------------------------------------
    # Shared chunking logic (splits into sentences)
    # ---------------------------------------------
    def _chunk_text(self, text: str, start_chunk_id: int):
        chunks = []
        chunk_id = start_chunk_id

        sentences = self._split_into_sentences(text)
        current_text = ""
        current_tokens = 0

        for sentence in sentences:
            sentence = sentence.strip()
            tok = self.count_tokens(sentence)

            # Oversized sentence
            if tok > self.max_tokens:
                # finalize current chunk if needed
                if current_tokens >= self.min_tokens:
                    chunks.append({"chunk_id": chunk_id, "text": current_text.strip()})
                    chunk_id += 1
                    current_text = ""
                    current_tokens = 0

                # add sentence as its own chunk
                chunks.append({"chunk_id": chunk_id, "text": sentence})
                chunk_id += 1
                continue

            # If adding sentence stays under max_tokens, do it
            if current_tokens + tok <= self.max_tokens:
                if current_text:
                    current_text += " " + sentence
                else:
                    current_text = sentence

                current_tokens += tok

            else:
                # finalize current chunk
                chunks.append({"chunk_id": chunk_id, "text": current_text.strip()})
                chunk_id += 1

                # start new chunk
                current_text = sentence
                current_tokens = tok

        # leftover chunk
        if current_text:
            chunks.append({"chunk_id": chunk_id, "text": current_text.strip()})

        return chunks

    # ---------------------------------------------
    # Special handling for multiple reviews
    # ---------------------------------------------
    def _chunk_reviews(self, reviews: List[str], start_chunk_id: int):
        chunks = []
        chunk_id = start_chunk_id
        current_text = ""
        current_tokens = 0

        for review in reviews:
            review = review.strip()
            if not review:
                continue

            tok = self.count_tokens(review)

            # Large review → chunk with sentence split
            if tok > self.max_tokens:
                # flush current chunk
                if current_text:
                    chunks.append({"chunk_id": chunk_id, "text": current_text.strip()})
                    chunk_id += 1
                    current_text = ""
                    current_tokens = 0

                # split this review
                sub_chunks = self._chunk_text(review, chunk_id)
                chunks.extend(sub_chunks)
                chunk_id += len(sub_chunks)
                continue

            # If fits in current chunk
            if current_tokens + tok <= self.max_tokens:
                if current_text:
                    current_text += "\n\n" + review
                else:
                    current_text = review

                current_tokens += tok

            else:
                # finalize current
                chunks.append({"chunk_id": chunk_id, "text": current_text.strip()})
                chunk_id += 1

                # start new chunk
                current_text = review
                current_tokens = tok

        # leftover reviews
        if current_text:
            chunks.append({"chunk_id": chunk_id, "text": current_text.strip()})

        return chunks

    # ---------------------------------------------
    # Sentence splitting
    # ---------------------------------------------
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Simple but robust sentence splitter.
        Does NOT split on ellipses or decimals.
        """

        if not text:
            return []

        sentences = []
        sentence = ""

        i = 0
        while i < len(text):
            char = text[i]
            sentence += char

            # check for typical sentence endings
            if char in [".", "?", "!"]:
                # prevent splitting on "..." (ellipsis)
                if i + 1 < len(text) and text[i:i+3] == "...":
                    i += 1
                    continue

                # prevent splitting decimals like 3.14
                if i > 0 and text[i-1].isdigit() and (i+1 < len(text) and text[i+1].isdigit()):
                    i += 1
                    continue

                # finalize sentence
                sentences.append(sentence.strip())
                sentence = ""

                # skip whitespace
                i += 1
                while i < len(text) and text[i].isspace():
                    i += 1
                continue

            i += 1

        if sentence.strip():
            sentences.append(sentence.strip())

        return sentences or [text]

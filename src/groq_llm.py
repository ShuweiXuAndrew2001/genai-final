"""
Groq LLM Module (FINAL WORKING VERSION)

Uses: llama-3.3-70b-versatile
Compatible with your Groq account and your pipeline structure.
"""

import os
import json
from groq import Groq
from dotenv import load_dotenv
from .prompts import PromptTemplates

load_dotenv()


class GroqLLM:
    """
    Wrapper around Groq's Chat Completions API.
    Provides summarization, JSON extraction, RAG refinement,
    visual attribute extraction, and image prompt generation.
    """

    def __init__(
        self,
        api_key=None,
        model: str = "llama-3.3-70b-versatile",   # ✅ YOUR supported model
        temperature: float = 0.2,
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Missing GROQ_API_KEY")

        self.client = Groq(api_key=self.api_key)
        self.model = model
        self.temperature = temperature

    # ---------------------------------------------------------
    # Low-level request function
    # ---------------------------------------------------------
    def _chat(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=self.temperature,
        )
        return response.choices[0].message.content.strip()

    # ---------------------------------------------------------
    # Summarization
    # ---------------------------------------------------------
    def summarize(self, text: str) -> str:
        prompt = PromptTemplates.get_summarization_prompt(text)
        return self._chat(prompt)

    # ---------------------------------------------------------
    # Extract structured JSON
    # ---------------------------------------------------------
    def extract_json(self, text: str) -> dict:
        prompt = PromptTemplates.get_json_extraction_prompt(text)
        raw = self._chat(prompt)

        cleaned = raw.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(cleaned)
        except Exception:
            return {"raw": raw, "error": "Invalid JSON"}

    # ---------------------------------------------------------
    # RAG refinement
    # ---------------------------------------------------------
    def refine_with_rag(self, query: str, retrieved_chunks):
        prompt = PromptTemplates.get_rag_refinement_prompt(query, retrieved_chunks)
        return self._chat(prompt)

    # ---------------------------------------------------------
    # Visual attributes
    # ---------------------------------------------------------
    def generate_visual_attributes(self, product_data: dict, analysis_summary: str) -> dict:
        prompt = PromptTemplates.get_visual_attributes_prompt(product_data, analysis_summary)
        raw = self._chat(prompt)

        cleaned = raw.replace("```json", "").replace("```", "").strip()

        try:
            return json.loads(cleaned)
        except Exception:
            return {"raw": raw, "error": "Invalid JSON"}

    # ---------------------------------------------------------
    # Final image prompt generation
    # ---------------------------------------------------------
    def generate_image_prompt(self, product_data: dict, visual_attributes: dict) -> str:
        prompt = PromptTemplates.get_image_prompt_prompt(product_data, visual_attributes)
        return self._chat(prompt)

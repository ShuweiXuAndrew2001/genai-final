"""
Prompt Templates Module

Provides a unified class (`PromptTemplates`) with structured prompt generators:
- Summarization
- Structured JSON extraction
- RAG refinement
- Visual attribute extraction
- Final diffusion model image prompt creation
"""

from typing import List, Dict, Any


class PromptTemplates:
    """
    Centralized, class-based prompt templates for all LLM operations.
    All methods are static and require no state.
    """

    # ----------------------------
    # 1. Summarization Prompt
    # ----------------------------
    @staticmethod
    def get_summarization_prompt(text: str) -> str:
        return f"""You are an expert product analyst. Summarize the following product information.

Product Information:
{text}

Provide a concise summary (3–4 sentences) that includes:
- Key product features
- Most important customer feedback themes
- Any visual details mentioned (colors, materials, shape)
- Overall sentiment tone

Summary:
"""

    # ----------------------------
    # 2. Structured JSON Extraction
    # ----------------------------
    @staticmethod
    def get_json_extraction_prompt(text: str) -> str:
        return f"""You are a data extraction specialist.
Extract structured product attributes from the following text and return ONLY valid JSON.

TEXT:
{text}

Return JSON with strictly this schema:
{{
    "product_name": "",
    "category": "",
    "key_features": [],
    "materials": [],
    "colors": [],
    "styles": [],
    "use_cases": [],
    "customer_sentiment": {{
        "positive_themes": [],
        "negative_themes": [],
        "overall_sentiment": "positive" | "neutral" | "negative"
    }}
}}
"""

    # ----------------------------
    # 3. RAG-Based Refinement Prompt
    # ----------------------------
    @staticmethod
    def get_rag_refinement_prompt(
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        max_chunks: int = 5
    ) -> str:
        context = "\n\n---\n\n".join([
            f"[Chunk {i+1} | Source: {chunk.get('source')}]\n{chunk.get('text')}"
            for i, chunk in enumerate(retrieved_chunks[:max_chunks])
        ])

        return f"""You are an expert product analyst. Use the context below to answer the question.

QUESTION:
{query}

CONTEXT:
{context}

Your answer must:
- Be factual
- Use specific evidence from the context
- Avoid hallucinating missing details
- Concisely summarize the correct information

ANSWER:
"""

    # ----------------------------
    # 4. Visual Attribute Extraction
    # ----------------------------
    @staticmethod
    def get_visual_attributes_prompt(
        product_data: Dict[str, Any],
        analysis_summary: str
    ) -> str:

        return f"""You are a senior visual design expert. Extract VISUAL attributes for use in a diffusion model.

PRODUCT NAME: {product_data.get('name', '')}
CATEGORY: {product_data.get('category', '')}

Combined Analysis:
{analysis_summary}

Extract visual attributes ONLY. Return valid JSON:

{{
    "primary_color": "",
    "secondary_colors": [],
    "material_texture": "",
    "surface_finish": "",
    "shape_form": "",
    "size_scale": "",
    "lighting_style": "",
    "background_style": "",
    "aesthetic_style": "",
    "key_visual_features": [],
    "brand_characteristics": []
}}

Return ONLY the JSON:
"""

    # ----------------------------
    # 5. Final Diffusion Image Prompt
    # ----------------------------
    @staticmethod
    def get_image_prompt_prompt(
        product_data: Dict[str, Any],
        visual_attributes: Dict[str, Any]
    ) -> str:

        attr_text = "\n".join([f"- {k}: {v}" for k, v in visual_attributes.items()])

        return f"""You are an expert prompt engineer for diffusion models (Stable Diffusion, Midjourney, DALL·E).
Create a highly detailed product photography prompt using the attributes below.

PRODUCT: {product_data.get('name')}
CATEGORY: {product_data.get('category')}

VISUAL ATTRIBUTES:
{attr_text}

Create a 2–3 sentence professional product image prompt that includes:
- Shape, color, material, texture
- Style, lighting, background
- Camera angle & composition
- High-quality photographic descriptors

Write ONLY the final diffusion prompt:
"""

    # ----------------------------
    # 6. Optional: Attribute Refinement for Color/Material
    # ----------------------------
    @staticmethod
    def get_attribute_refinement_prompt(attribute: str, text: str) -> str:
        return f"""Based on the text below, determine the most consistent description for the attribute: {attribute}.

TEXT:
{text}

Return a short, factual statement with NO hallucinations.
"""



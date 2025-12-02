# pipeline.py
from typing import Dict, Any, List
from prompts import PromptTemplates  # or 'prompts'
from openai import OpenAI
from dotenv import load_dotenv
import google.generativeai as genai   # pip install google-generativeai
import json
import base64
import os
import uuid

client = OpenAI()
load_dotenv()   # this loads .env file

# now you can read the key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def call_llm(prompt: str) -> str:
    response = client.responses.create(
        model="gpt-5",
        input=prompt
    )
    return response.output[0].content[0].text

def call_openai_image(prompt: str, model_name: str) -> str:
    result = client_openai.images.generate(
        model=model_name,
        prompt=prompt,
        size="1024x1024"
    )
    image_b64 = result.data[0].b64_json
    image_bytes = base64.b64decode(image_b64)

    filename = f"openai_{model_name}_{uuid.uuid4().hex}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(image_bytes)
    return filepath

def call_gemini_image(prompt: str, model_name: str) -> str:
    """
    Example Gemini image call. Adjust to your actual Gemini API usage.
    """
    model = genai.GenerativeModel(model_name)
    resp = model.generate_image(prompt=prompt)  # this line is pseudocode; check actual API
    image_bytes = resp.image  # adapt based on real response

    filename = f"gemini_{model_name}_{uuid.uuid4().hex}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(image_bytes)
    return filepath


def call_image_model(prompt: str, model_name: str) -> str:
    if model_name.startswith("openai:"):
        openai_model = model_name.split(":", 1)[1]
        return call_openai_image(prompt, openai_model)

    if model_name.startswith("gemini:"):
        gemini_model = model_name.split(":", 1)[1]
        return call_gemini_image(prompt, gemini_model)


def apply_manual_prompt_fixes(product: Dict[str, Any], prompt: str) -> str:
    pid = product.get("id")

    if pid == "product_1_razer":
        prompt = prompt.replace(
            "kitty ear design with the whole product in white accents",
            "kitty ear design with the whole product in white accents, the ear contains quartz pink RGB light"
        )
        if "boomless and wireless" not in prompt:
            prompt += " This product is boomless and wireless."
    elif pid == "product_3_crocs":
        prompt = prompt.replace("accents of brown", "")
        prompt = prompt.replace(
            "pivoting heel straps",
            "pivoting heel straps with only the connected spot in black"
        )

    return prompt


def run_text_pipeline_for_product(product: Dict[str, Any]) -> Dict[str, Any]:
    raw_text = product["description"] + "\n\n" + "\n".join(product["reviews"])
    summ_prompt = PromptTemplates.get_summarization_prompt(raw_text)
    analysis_summary = call_llm(summ_prompt)

    attr_prompt = PromptTemplates.get_visual_attributes_prompt(
        product_data=product,
        analysis_summary=analysis_summary
    )
    attr_json_text = call_llm(attr_prompt)
    visual_attributes = json.loads(attr_json_text)

    final_prompt_prompt = PromptTemplates.get_image_prompt_prompt(
        product_data=product,
        visual_attributes=visual_attributes
    )
    diffusion_prompt = call_llm(final_prompt_prompt)
    diffusion_prompt = apply_manual_prompt_fixes(product, diffusion_prompt)

    return {
        "analysis_summary": analysis_summary,
        "visual_attributes": visual_attributes,
        "diffusion_prompt": diffusion_prompt,
    }


def run_image_generation_for_product(
    product: Dict[str, Any],
    models: List[str]
) -> List[Dict[str, Any]]:
    text_outputs = run_text_pipeline_for_product(product)
    prompt = text_outputs["diffusion_prompt"]

    results = []
    for model_name in models:
        image_path = call_image_model(prompt=prompt, model_name=model_name)

        results.append({
            "product_id": product["id"],
            "product_name": product["name"],
            "model": model_name,
            "prompt": prompt,
            "image_path": image_path,
            "analysis_summary": text_outputs["analysis_summary"],
            "visual_attributes": text_outputs["visual_attributes"],
        })

    return results


def run_pipeline_for_all_products(products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    all_results = []
    models = ["openai:gpt-image-1", "gemini:gemini-2.0-flash-image"]

    for product in products:
        results = run_image_generation_for_product(product, models=models)
        all_results.extend(results)

    return all_results

"""
Main Entry Point

Run the full Q2 pipeline:
- Load products JSON
- Chunk + embed + index
- Run RAG + LLM analysis
- Generate visual attributes + diffusion prompts
"""

import os
import json
from dotenv import load_dotenv
from src.analyzer import ProductAnalyzer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

def main():
    data_path = os.path.join(BASE_DIR, "data", "products_data.json")
    output_dir = os.path.join(BASE_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)

    print(f"📥 Loading products from {data_path}...")
    with open(data_path, "r", encoding="utf-8") as f:
        products = json.load(f)
    print(f"→ Loaded {len(products)} products")

    analyzer = ProductAnalyzer()

    print("🔧 Building embeddings + FAISS index...")
    analyzer.process_products(products)

    print("🎯 Running full analysis on all products...")
    results = analyzer.analyze_all_products(
        products,
        output_dir=output_dir
    )

    results_path = os.path.join(output_dir, "analysis_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n✅ Analysis complete!")
    print(f"📄 Results saved to: {results_path}")
    print(f"📁 Extracted JSON: {os.path.join(output_dir, 'extracted')}")
    print(f"🖼  Image prompts:  {os.path.join(output_dir, 'prompts')}")

    if results:
        first = results[0]
        print("\n--- Sample Output ---")
        print("Product:", first.get("product_name"))
        print("Summary:", first.get("summary", "")[:200], "...")
        print("Image Prompt:", first.get("image_prompt", "")[:200], "...")

if __name__ == "__main__":
    main()

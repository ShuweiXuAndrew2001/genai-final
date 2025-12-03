"""Agentic workflow orchestrating Q2 analysis and Q3 image generation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv

from .analyzer import ProductAnalyzer
from .image_api_call import apply_manual_prompt_fixes, call_image_model

load_dotenv()

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = BASE_DIR / "data" / "products_data.json"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs"


@dataclass
class WorkflowConfig:
    data_path: Path = DEFAULT_DATA_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
    generate_images: bool = True
    image_models: List[str] = field(
        default_factory=lambda: [
            "openai:gpt-image-1",
            "gemini:gemini-2.0-flash-image",
        ]
    )
    save_combined_results: bool = True
    run_label: Optional[str] = None


class AgenticWorkflow:
    def __init__(
        self,
        config: WorkflowConfig,
        analyzer_factory: Optional[Callable[[], ProductAnalyzer]] = None,
    ) -> None:
        self.config = config
        self._analyzer_factory = analyzer_factory or ProductAnalyzer
        self._run_dir: Optional[Path] = None

    def run(self) -> Dict[str, Any]:
        self._prepare_logging()
        logger.info("Starting agentic workflow")
        products = self._load_products()
        run_dir = self._prepare_run_directory()

        analyzer = self._analyzer_factory()
        q2_output_dir = run_dir / "q2_analysis"
        q2_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Processing products through Q2 pipeline")
        analyzer.process_products(products)
        analysis_results = analyzer.analyze_all_products(
            products,
            output_dir=str(q2_output_dir),
        )
        analysis_by_id = {res.get("product_id"): res for res in analysis_results}

        combined_products: List[Dict[str, Any]] = []
        for product in products:
            product_id = product.get("id")
            analysis = analysis_by_id.get(product_id, {})
            entry: Dict[str, Any] = {
                "product_metadata": {
                    "id": product_id,
                    "name": product.get("name"),
                    "category": product.get("category"),
                    "source_link": product.get("link"),
                    "real_image_path": self._resolve_path(product.get("real_image_path")),
                },
                "analysis": {
                    "summary": analysis.get("summary"),
                    "structured_data": analysis.get("structured_data"),
                    "rag_analysis": analysis.get("rag_analysis"),
                    "visual_attributes": analysis.get("visual_attributes"),
                    "image_prompt": analysis.get("image_prompt"),
                    "artifact_dir": str(q2_output_dir),
                },
            }

            if self.config.generate_images:
                entry["image_generation"] = self._generate_images_for_product(
                    product,
                    analysis,
                    run_dir,
                )

            combined_products.append(entry)

        final_payload = {
            "metadata": {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "run_label": self.config.run_label,
                "config": {
                    "data_path": str(self.config.data_path),
                    "output_dir": str(run_dir),
                    "generate_images": self.config.generate_images,
                    "image_models": self.config.image_models,
                },
                "counts": {
                    "products": len(products),
                    "analysis_records": len(analysis_results),
                },
            },
            "products": combined_products,
        }

        if self.config.save_combined_results:
            output_path = run_dir / "agentic_results.json"
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(final_payload, f, indent=2, ensure_ascii=False)
            logger.info("Saved combined workflow results to %s", output_path)

        return final_payload

    def _load_products(self) -> List[Dict[str, Any]]:
        if not self.config.data_path.exists():
            raise FileNotFoundError(f"Product data not found at {self.config.data_path}")
        with self.config.data_path.open("r", encoding="utf-8") as f:
            products = json.load(f)
        logger.info("Loaded %d products", len(products))
        return products

    def _prepare_run_directory(self) -> Path:
        if self._run_dir is not None:
            return self._run_dir
        run_stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        if self.config.run_label:
            run_name = f"{run_stamp}_{self.config.run_label}"
        else:
            run_name = run_stamp
        run_dir = self.config.output_dir / "agentic_runs" / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        self._run_dir = run_dir
        logger.info("Run directory prepared at %s", run_dir)
        return run_dir

    def _generate_images_for_product(
        self,
        product: Dict[str, Any],
        analysis: Dict[str, Any],
        run_dir: Path,
    ) -> Dict[str, Any]:
        prompt = (analysis or {}).get("image_prompt")
        if not prompt:
            return {
                "prompt": None,
                "generated": [],
                "errors": ["Missing image prompt from Q2 analysis"],
            }

        adjusted_prompt = apply_manual_prompt_fixes(product, prompt)
        results: List[Dict[str, Any]] = []
        errors: List[str] = []
        images_dir = run_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        for model in self.config.image_models:
            try:
                image_path = call_image_model(adjusted_prompt, model)
                results.append(
                    {
                        "model": model,
                        "image_path": image_path,
                    }
                )
            except Exception as exc:
                errors.append(f"{model}: {exc}")
                logger.warning("Image generation failed for %s: %s", model, exc)

        comparison = {
            "real_image_path": self._resolve_path(product.get("real_image_path")),
            "generated": results,
            "errors": errors,
            "prompt": adjusted_prompt,
        }
        return comparison

    def _resolve_path(self, relative_path: Optional[str]) -> Optional[str]:
        if not relative_path:
            return None
        path = Path(relative_path)
        if not path.is_absolute():
            path = BASE_DIR / path
        return str(path)

    def _prepare_logging(self) -> None:
        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO)
        logger.setLevel(logging.INFO)


def run_agentic_workflow(config: Optional[WorkflowConfig] = None) -> Dict[str, Any]:
    workflow = AgenticWorkflow(config or WorkflowConfig())
    return workflow.run()


if __name__ == "__main__":
    run_agentic_workflow()
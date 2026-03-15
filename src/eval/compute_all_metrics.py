"""Script de calcul de toutes les métriques sur un répertoire de runs."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import jsonlines
import pandas as pd

from src.eval.qa_metrics import QAMetrics
from src.eval.retrieval_metrics import RetrievalMetrics
from src.eval.statistical_tests import StatisticalTests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)


def load_run_results(run_dir: str) -> list[dict]:
    """Charge tous les fichiers JSONL d'un répertoire de run."""
    results = []
    for path in sorted(Path(run_dir).glob("*.jsonl")):
        with jsonlines.open(path) as reader:
            for item in reader:
                results.append(item)
    logger.info("Chargé %d résultats depuis %s", len(results), run_dir)
    return results


def compute_metrics_for_run(results: list[dict]) -> pd.DataFrame:
    """Calcule les métriques agrégées par configuration."""
    qa = QAMetrics()
    rows = []

    configs = {}
    for item in results:
        config_key = json.dumps(item.get("config", {}), sort_keys=True)
        configs.setdefault(config_key, []).append(item)

    for config_key, items in configs.items():
        config = json.loads(config_key)
        predictions = [i.get("generated_answer", "") for i in items]
        gold_answers = [i.get("gold_answer", []) for i in items]
        gold_answers = [[g] if isinstance(g, str) else g for g in gold_answers]

        metrics = qa.compute_batch(predictions, gold_answers)

        faithfulness_scores = [
            i.get("metrics", {}).get("faithfulness", 0.0) for i in items
            if i.get("metrics", {}).get("faithfulness") is not None
        ]
        avg_faith = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else None

        recall_scores = [
            i.get("metrics", {}).get("recall_at_10", 0.0) for i in items
            if i.get("metrics", {}).get("recall_at_10") is not None
        ]
        avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else None

        row = {
            **config,
            "n_samples": len(items),
            "em": round(metrics["em"], 4),
            "f1": round(metrics["f1"], 4),
        }
        if avg_faith is not None:
            row["faithfulness"] = round(avg_faith, 4)
        if avg_recall is not None:
            row["recall_at_10"] = round(avg_recall, 4)

        rows.append(row)

    return pd.DataFrame(rows)


def main(run_dir: str, output_dir: str = "reports/tables/") -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = load_run_results(run_dir)

    if not results:
        logger.warning("Aucun résultat trouvé dans %s", run_dir)
        return

    df = compute_metrics_for_run(results)

    run_name = Path(run_dir).name
    output_path = Path(output_dir) / f"metrics_{run_name}.csv"
    df.to_csv(output_path, index=False)
    logger.info("Métriques sauvegardées → %s", output_path)

    logger.info("\n%s", df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calcul des métriques RAG")
    parser.add_argument("--run", required=True, help="Répertoire des runs JSONL")
    parser.add_argument("--output", default="reports/tables/", help="Répertoire de sortie CSV")
    args = parser.parse_args()
    main(args.run, args.output)

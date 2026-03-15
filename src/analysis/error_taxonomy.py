"""Classification des erreurs RAG selon la taxonomie du mémoire."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """
    Taxonomie des erreurs RAG.

    Basée sur l'annotation manuelle de 120 cas (FR + EN).
    """

    RETRIEVAL_FAILURE = "retrieval_failure"
    """Passage pertinent absent du top-k (~35%). Signal : Recall@k < 0.5"""

    CONTEXT_NOISE_SATURATION = "context_noise_saturation"
    """Distracteurs dominants → dérive de réponse (~25%). Signal : Faithfulness < 0.6 avec Recall@k élevé"""

    GROUNDING_FAILURE = "grounding_failure"
    """Hallucination malgré preuve correcte (~20%). Signal : Faithfulness faible, EM correct"""

    LOST_IN_THE_MIDDLE = "lost_in_the_middle"
    """Preuve présente mais ignorée (position) (~10%). Signal : perf. dégradée selon position"""

    INTEGRATION_FAILURE = "integration_failure"
    """Multi-preuves mal combinées (~5%). Signal : Multi-hop F1 << Single-hop F1"""

    UNANSWERABLE_MISHANDLING = "unanswerable_mishandling"
    """Réponse inventée au lieu d'abstention (~5%). Signal : hallucination > 50% sur unanswerable"""

    UNKNOWN = "unknown"


ERROR_THRESHOLDS = {
    ErrorType.RETRIEVAL_FAILURE: {"recall_at_10": 0.5},
    ErrorType.CONTEXT_NOISE_SATURATION: {"faithfulness": 0.6, "recall_at_10": 0.7},
    ErrorType.GROUNDING_FAILURE: {"faithfulness": 0.6},
    ErrorType.UNANSWERABLE_MISHANDLING: {"is_unanswerable": True},
}

ERROR_CORRECTIONS = {
    ErrorType.RETRIEVAL_FAILURE: "Retrieval hybride, fine-tuning encodeur",
    ErrorType.CONTEXT_NOISE_SATURATION: "Reranking, réduire k, filtrer doublons",
    ErrorType.GROUNDING_FAILURE: "Format citations strictes, instruction abstention",
    ErrorType.LOST_IN_THE_MIDDLE: "Format citations, passage gold en premier",
    ErrorType.INTEGRATION_FAILURE: "GraphRAG, RAPTOR, augmenter k avec reranking",
    ErrorType.UNANSWERABLE_MISHANDLING: "Format avec instruction abstention explicite",
}


@dataclass
class AnnotatedCase:
    """Un cas annoté avec son type d'erreur."""

    question_id: str
    question: str
    gold_answer: str
    generated_answer: str
    config: dict[str, Any]
    error_type: ErrorType
    metrics: dict[str, float] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "gold_answer": self.gold_answer,
            "generated_answer": self.generated_answer,
            "config": self.config,
            "error_type": self.error_type.value,
            "metrics": self.metrics,
            "notes": self.notes,
        }


class ErrorTaxonomy:
    """
    Classificateur automatique des erreurs RAG.

    Basé sur des règles heuristiques à partir des métriques disponibles.
    Conçu pour l'annotation préliminaire des 120 cas qualitatifs.
    """

    def classify(self, item: dict[str, Any]) -> ErrorType:
        """
        Classe automatiquement le type d'erreur d'un exemple.

        Args:
            item: résultat de run avec métriques (format JSONL standard)

        Returns:
            Type d'erreur identifié
        """
        metrics = item.get("metrics", {})
        config = item.get("config", {})

        em = metrics.get("em", 0)
        f1 = metrics.get("f1", 0.0)
        recall = metrics.get("recall_at_10", 1.0)
        faithfulness = metrics.get("faithfulness", 1.0)
        is_unanswerable = item.get("is_unanswerable", False)
        generated = item.get("generated_answer", "")

        if is_unanswerable and "je ne sais pas" not in generated.lower():
            return ErrorType.UNANSWERABLE_MISHANDLING

        if recall < 0.5 and em == 0:
            return ErrorType.RETRIEVAL_FAILURE

        if recall >= 0.7 and faithfulness < 0.6 and em == 0:
            return ErrorType.CONTEXT_NOISE_SATURATION

        if faithfulness < 0.6 and em == 1:
            return ErrorType.GROUNDING_FAILURE

        if em == 1 or f1 >= 0.7:
            return ErrorType.UNKNOWN

        return ErrorType.GROUNDING_FAILURE

    def classify_batch(self, items: list[dict[str, Any]]) -> list[ErrorType]:
        """Classe un lot d'exemples."""
        return [self.classify(item) for item in items]

    def compute_distribution(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Calcule la distribution des types d'erreurs sur un corpus.

        Args:
            items: résultats de run avec métriques

        Returns:
            Distribution des erreurs avec fréquences et pourcentages
        """
        error_types = self.classify_batch(items)
        counts: dict[str, int] = {}
        for et in error_types:
            counts[et.value] = counts.get(et.value, 0) + 1

        n = len(items)
        distribution = {
            et_val: {
                "count": count,
                "percentage": round(100 * count / n, 1),
                "correction": ERROR_CORRECTIONS.get(ErrorType(et_val), ""),
            }
            for et_val, count in sorted(counts.items(), key=lambda x: x[1], reverse=True)
        }

        return {"total": n, "distribution": distribution}

    def save_annotated_cases(
        self,
        cases: list[AnnotatedCase],
        output_path: str = "annexes/qualitative_cases/annotated_120.jsonl",
    ) -> None:
        """Sauvegarde les cas annotés au format JSONL."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for case in cases:
                f.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")
        logger.info("Sauvegardé %d cas annotés → %s", len(cases), output_path)

    def sample_for_annotation(
        self,
        items: list[dict[str, Any]],
        n_per_quartile: int = 30,
        random_seed: int = 42,
    ) -> list[dict[str, Any]]:
        """
        Sélectionne 120 cas pour l'annotation manuelle.

        Stratégie : 30 cas par quartile de performance (F1).

        Args:
            items: tous les résultats disponibles
            n_per_quartile: nombre de cas par quartile (défaut 30 → 120 total)
            random_seed: graine pour la reproductibilité

        Returns:
            120 cas représentatifs pour l'annotation
        """
        import random
        rng = random.Random(random_seed)

        sorted_items = sorted(items, key=lambda x: x.get("metrics", {}).get("f1", 0.0))
        n = len(sorted_items)
        q_size = n // 4

        sampled = []
        for q in range(4):
            start = q * q_size
            end = (q + 1) * q_size if q < 3 else n
            quartile = sorted_items[start:end]
            sample = rng.sample(quartile, min(n_per_quartile, len(quartile)))
            sampled.extend(sample)

        return sampled

"""Métriques de retrieval : Recall@k, MRR, nDCG via pytrec_eval."""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

# pytrec_eval est utilisé uniquement dans compute_with_pytrec_eval (lazy import)


class RetrievalMetrics:
    """
    Calcule les métriques standard de retrieval d'information.

    Métriques : Recall@k, MRR, nDCG@k
    Utilise pytrec_eval pour la compatibilité avec les benchmarks TREC.
    """

    @staticmethod
    def recall_at_k(
        retrieved_ids: list[str],
        relevant_ids: set[str],
        k: int,
    ) -> float:
        """
        Calcule le Recall@k.

        Args:
            retrieved_ids: identifiants des passages récupérés (ordonnés par rang)
            relevant_ids: ensemble des passages pertinents (gold)
            k: seuil de coupure

        Returns:
            Recall@k ∈ [0, 1]
        """
        if not relevant_ids:
            return 0.0
        top_k = set(retrieved_ids[:k])
        return len(top_k & relevant_ids) / len(relevant_ids)

    @staticmethod
    def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
        """
        Calcule le Mean Reciprocal Rank (MRR).

        Args:
            retrieved_ids: identifiants des passages récupérés
            relevant_ids: ensemble des passages pertinents

        Returns:
            MRR ∈ [0, 1]
        """
        for rank, pid in enumerate(retrieved_ids, start=1):
            if pid in relevant_ids:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def ndcg_at_k(
        retrieved_ids: list[str],
        relevance_scores: dict[str, float],
        k: int,
    ) -> float:
        """
        Calcule le nDCG@k (Normalized Discounted Cumulative Gain).

        Args:
            retrieved_ids: identifiants récupérés (ordonnés par rang)
            relevance_scores: dictionnaire {passage_id: score_pertinence}
            k: seuil de coupure

        Returns:
            nDCG@k ∈ [0, 1]
        """
        def dcg(ids: list[str]) -> float:
            gain = 0.0
            for i, pid in enumerate(ids[:k], start=1):
                rel = relevance_scores.get(pid, 0.0)
                gain += rel / math.log2(i + 1)
            return gain

        actual_dcg = dcg(retrieved_ids)

        ideal_ids = sorted(relevance_scores, key=relevance_scores.get, reverse=True)
        ideal_dcg = dcg(ideal_ids)

        return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    def compute_batch(
        self,
        retrieved_results: list[list[str]],
        relevant_sets: list[set[str]],
        k_values: list[int] | None = None,
    ) -> dict[str, float]:
        """
        Calcule les métriques moyennes sur un lot de requêtes.

        Args:
            retrieved_results: liste de listes d'identifiants récupérés
            relevant_sets: liste d'ensembles de passages pertinents
            k_values: valeurs de k à évaluer (défaut : [5, 10, 20])

        Returns:
            Dictionnaire de métriques agrégées
        """
        k_values = k_values or [5, 10, 20]
        metrics: dict[str, list[float]] = {f"recall_at_{k}": [] for k in k_values}
        metrics["mrr"] = []

        for retrieved, relevant in zip(retrieved_results, relevant_sets):
            for k in k_values:
                metrics[f"recall_at_{k}"].append(self.recall_at_k(retrieved, relevant, k))
            metrics["mrr"].append(self.mrr(retrieved, relevant))

        return {key: sum(vals) / len(vals) if vals else 0.0 for key, vals in metrics.items()}

    def compute_with_pytrec_eval(
        self,
        run: dict[str, dict[str, float]],
        qrels: dict[str, dict[str, int]],
        measures: set[str] | None = None,
    ) -> dict[str, Any]:
        """
        Calcule les métriques via pytrec_eval (format TREC standard).

        Args:
            run: {query_id: {doc_id: score}}
            qrels: {query_id: {doc_id: pertinence}}
            measures: métriques à calculer (défaut : recall, ndcg, recip_rank)

        Returns:
            Métriques agrégées
        """
        try:
            import pytrec_eval
        except ImportError as e:
            raise ImportError("pytrec_eval requis. pip install pytrec_eval") from e

        measures = measures or {"recall_10", "ndcg_cut_10", "recip_rank"}
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, measures)
        results = evaluator.evaluate(run)

        aggregated: dict[str, list[float]] = {}
        for qid, scores in results.items():
            for measure, value in scores.items():
                aggregated.setdefault(measure, []).append(value)

        return {m: sum(vs) / len(vs) for m, vs in aggregated.items()}

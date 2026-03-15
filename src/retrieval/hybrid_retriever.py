"""Retriever hybride : fusion linéaire des scores BM25 et dense."""

from __future__ import annotations

import logging

from .bm25_retriever import BM25Retriever, RetrievedPassage
from .dense_retriever import DenseRetriever

logger = logging.getLogger(__name__)


def normalize_scores(passages: list[RetrievedPassage], score_type: str) -> dict[str, float]:
    """
    Normalise les scores d'une liste de passages en [0, 1].

    Args:
        passages: passages avec scores bruts
        score_type: 'bm25' ou 'dense'

    Returns:
        Dictionnaire {passage_id: score_normalisé}
    """
    scores = {
        p.passage_id: (p.bm25_score if score_type == "bm25" else p.dense_score)
        for p in passages
    }
    if not scores:
        return {}

    min_s = min(scores.values())
    max_s = max(scores.values())
    denom = max_s - min_s if max_s != min_s else 1.0

    return {pid: (s - min_s) / denom for pid, s in scores.items()}


class HybridRetriever:
    """
    Retriever hybride combinant BM25 et dense via fusion linéaire.

    score_final = alpha * score_dense + (1 - alpha) * score_bm25

    alpha retenu : 0.5
    alpha testé : 0.3, 0.5, 0.7
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        alpha: float = 0.5,
    ) -> None:
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.alpha = alpha

    def hybrid_score(self, bm25_score: float, dense_score: float) -> float:
        """
        Calcule le score hybride par fusion linéaire.

        Args:
            bm25_score: score BM25 normalisé
            dense_score: score dense normalisé

        Returns:
            Score hybride fusionné
        """
        return self.alpha * dense_score + (1 - self.alpha) * bm25_score

    def retrieve(self, query: str, k: int = 10, initial_k: int | None = None) -> list[RetrievedPassage]:
        """
        Récupère les top-k passages par fusion linéaire BM25 + Dense.

        Args:
            query: question en langage naturel
            k: nombre final de passages à retourner
            initial_k: nombre de candidats par retriever (défaut : max(k*3, 50))

        Returns:
            Liste de passages classés par score hybride décroissant
        """
        initial_k = initial_k or max(k * 3, 50)

        bm25_results = self.bm25.retrieve(query, k=initial_k)
        dense_results = self.dense.retrieve(query, k=initial_k)

        bm25_scores_norm = normalize_scores(bm25_results, "bm25")
        dense_scores_norm = normalize_scores(dense_results, "dense")

        all_ids = set(bm25_scores_norm) | set(dense_scores_norm)

        passage_index: dict[str, RetrievedPassage] = {}
        for p in bm25_results + dense_results:
            if p.passage_id not in passage_index:
                passage_index[p.passage_id] = p

        fused = []
        for pid in all_ids:
            b_score = bm25_scores_norm.get(pid, 0.0)
            d_score = dense_scores_norm.get(pid, 0.0)
            h_score = self.hybrid_score(b_score, d_score)
            passage = passage_index[pid]
            fused.append(RetrievedPassage(
                rank=0,
                passage_id=pid,
                text=passage.text,
                bm25_score=passage.bm25_score,
                dense_score=passage.dense_score,
                hybrid_score=h_score,
                metadata=passage.metadata,
            ))

        fused.sort(key=lambda p: p.hybrid_score, reverse=True)
        for rank, p in enumerate(fused[:k], start=1):
            p.rank = rank

        return fused[:k]

    def retrieve_batch(self, queries: list[str], k: int = 10) -> list[list[RetrievedPassage]]:
        """Récupère les top-k passages pour un lot de requêtes."""
        return [self.retrieve(q, k) for q in queries]

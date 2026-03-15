"""Rerankers : ColBERT (late interaction) et Cross-encodeur."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from .bm25_retriever import RetrievedPassage

logger = logging.getLogger(__name__)


class BaseReranker(ABC):
    """Interface commune pour tous les rerankers."""

    @abstractmethod
    def rerank(self, query: str, passages: list[RetrievedPassage], top_k: int = 10) -> list[RetrievedPassage]:
        """
        Reclasse les passages selon leur pertinence pour la requête.

        Args:
            query: question en langage naturel
            passages: passages candidats (généralement top-50)
            top_k: nombre de passages à conserver après reranking

        Returns:
            Passages reclassés (top_k passages)
        """


class ColBERTReranker(BaseReranker):
    """
    Reranker ColBERT via la bibliothèque RAGatouille.

    Utilise l'interaction tardive (MaxSim) entre tokens de la requête
    et tokens des passages. Plus rapide que le cross-encodeur.
    """

    def __init__(self, model_name: str = "colbert-ir/colbertv2.0") -> None:
        self.model_name = model_name
        self._model = None

    def _load_model(self) -> None:
        try:
            from ragatouille import RAGPretrainedModel
            self._model = RAGPretrainedModel.from_pretrained(self.model_name)
            logger.info("ColBERT chargé : %s", self.model_name)
        except ImportError as e:
            raise ImportError("ragatouille requis. pip install ragatouille") from e

    def rerank(self, query: str, passages: list[RetrievedPassage], top_k: int = 10) -> list[RetrievedPassage]:
        """Reclasse les passages avec ColBERT (MaxSim)."""
        if self._model is None:
            self._load_model()

        texts = [p.text for p in passages]
        passage_by_text = {p.text: p for p in passages}

        results = self._model.rerank(query=query, documents=texts, k=top_k)

        reranked = []
        for rank, result in enumerate(results, start=1):
            text = result["content"]
            original = passage_by_text.get(text)
            if original:
                reranked_passage = RetrievedPassage(
                    rank=rank,
                    passage_id=original.passage_id,
                    text=original.text,
                    bm25_score=original.bm25_score,
                    dense_score=original.dense_score,
                    hybrid_score=original.hybrid_score,
                    rerank_score=float(result.get("score", 0.0)),
                    metadata=original.metadata,
                )
                reranked.append(reranked_passage)

        return reranked[:top_k]


class CrossEncoderReranker(BaseReranker):
    """
    Reranker Cross-encodeur via sentence-transformers.

    Traite la requête et le document simultanément.
    Plus précis que ColBERT mais plus lent.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self._model = None

    def _load_model(self) -> None:
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            logger.info("Cross-encodeur chargé : %s", self.model_name)
        except ImportError as e:
            raise ImportError("sentence-transformers requis. pip install sentence-transformers") from e

    def rerank(self, query: str, passages: list[RetrievedPassage], top_k: int = 10) -> list[RetrievedPassage]:
        """Reclasse les passages avec le cross-encodeur."""
        if self._model is None:
            self._load_model()

        pairs = [(query, p.text) for p in passages]
        scores = self._model.predict(pairs)

        scored = list(zip(scores, passages))
        scored.sort(key=lambda x: x[0], reverse=True)

        reranked = []
        for rank, (score, original) in enumerate(scored[:top_k], start=1):
            reranked_passage = RetrievedPassage(
                rank=rank,
                passage_id=original.passage_id,
                text=original.text,
                bm25_score=original.bm25_score,
                dense_score=original.dense_score,
                hybrid_score=original.hybrid_score,
                rerank_score=float(score),
                metadata=original.metadata,
            )
            reranked.append(reranked_passage)

        return reranked


def get_reranker(reranker_type: str = "colbert") -> BaseReranker:
    """
    Factory pour instancier le bon reranker.

    Args:
        reranker_type: 'colbert' ou 'cross_encoder'

    Returns:
        Instance du reranker demandé
    """
    if reranker_type == "colbert":
        return ColBERTReranker()
    if reranker_type == "cross_encoder":
        return CrossEncoderReranker()
    raise ValueError(f"Reranker inconnu : {reranker_type}. Choisir 'colbert' ou 'cross_encoder'.")

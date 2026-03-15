"""Retriever BM25 via Pyserini / Lucene."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RetrievedPassage:
    """Passage récupéré par un retriever."""

    rank: int
    passage_id: str
    text: str
    bm25_score: float = 0.0
    dense_score: float = 0.0
    hybrid_score: float = 0.0
    rerank_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "passage_id": self.passage_id,
            "text": self.text,
            "bm25_score": self.bm25_score,
            "dense_score": self.dense_score,
            "hybrid_score": self.hybrid_score,
            "rerank_score": self.rerank_score,
            "metadata": self.metadata,
        }


class BM25Retriever:
    """
    Retriever BM25 utilisant Pyserini (wrapper Lucene).

    Hyperparamètres retenus : k1=1.0, b=0.6
    """

    def __init__(
        self,
        index_dir: str = "indexes/bm25/",
        k1: float = 1.0,
        b: float = 0.6,
    ) -> None:
        self.index_dir = index_dir
        self.k1 = k1
        self.b = b
        self._searcher = None

    def _load_searcher(self) -> None:
        """Charge le searcher Pyserini (lazy loading)."""
        try:
            from pyserini.search.lucene import LuceneSearcher
            self._searcher = LuceneSearcher(self.index_dir)
            self._searcher.set_bm25(self.k1, self.b)
            logger.info("Index BM25 chargé depuis %s (k1=%.1f, b=%.2f)", self.index_dir, self.k1, self.b)
        except ImportError as e:
            raise ImportError("pyserini requis pour BM25Retriever. pip install pyserini") from e

    def retrieve(self, query: str, k: int = 10) -> list[RetrievedPassage]:
        """
        Récupère les top-k passages pour une requête.

        Args:
            query: question en langage naturel
            k: nombre de passages à retourner

        Returns:
            Liste de passages classés par score BM25 décroissant
        """
        if self._searcher is None:
            self._load_searcher()

        hits = self._searcher.search(query, k=k)
        passages = []
        for rank, hit in enumerate(hits, start=1):
            raw = hit.raw
            import json
            try:
                doc = json.loads(raw)
                text = doc.get("contents", "")
            except Exception:
                text = raw or ""
            passages.append(RetrievedPassage(
                rank=rank,
                passage_id=hit.docid,
                text=text,
                bm25_score=float(hit.score),
            ))
        return passages

    def retrieve_batch(self, queries: list[str], k: int = 10) -> list[list[RetrievedPassage]]:
        """Récupère les top-k passages pour un lot de requêtes."""
        return [self.retrieve(q, k) for q in queries]

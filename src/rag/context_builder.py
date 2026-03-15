"""Assemblage du contexte pour le prompt RAG."""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.retrieval.bm25_retriever import RetrievedPassage

logger = logging.getLogger(__name__)

POSITIONS = ["first", "middle", "last"]


class ContextBuilder:
    """
    Assemble les passages récupérés en un contexte pour le prompt.

    Variables testées :
    - Ordre : pertinent en 1er / au milieu / en dernier
    - k : 5, 10, 20
    - Format : standard / citations
    """

    def build(
        self,
        passages: list[RetrievedPassage],
        gold_passage_id: str | None = None,
        position: str = "first",
    ) -> list[RetrievedPassage]:
        """
        Réorganise les passages selon la position du passage gold (Protocole B).

        Args:
            passages: passages récupérés (classés par score)
            gold_passage_id: identifiant du passage gold (si connu)
            position: 'first', 'middle', ou 'last'

        Returns:
            Passages réordonnés
        """
        if gold_passage_id is None or position == "first":
            return passages

        gold = [p for p in passages if p.passage_id == gold_passage_id]
        others = [p for p in passages if p.passage_id != gold_passage_id]

        if not gold:
            return passages

        k = len(passages)

        if position == "middle":
            mid = k // 2
            reordered = others[:mid] + gold + others[mid:]
        elif position == "last":
            reordered = others + gold
        else:
            reordered = passages

        for rank, p in enumerate(reordered, start=1):
            p.rank = rank

        return reordered

    def build_with_distractors(
        self,
        gold_passage: RetrievedPassage,
        distractor_passages: list[RetrievedPassage],
        k_total: int = 10,
        distractor_ratio: float = 0.4,
        position: str = "first",
        random_seed: int = 42,
    ) -> list[RetrievedPassage]:
        """
        Construit un contexte avec une proportion contrôlée de distracteurs (Protocole B).

        Args:
            gold_passage: passage contenant la réponse correcte
            distractor_passages: passages distracteurs (sans la réponse)
            k_total: nombre total de passages dans le contexte
            distractor_ratio: proportion de distracteurs p ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
            position: position du passage gold ('first', 'middle', 'last')
            random_seed: graine pour la reproductibilité

        Returns:
            Liste de k_total passages mélangés selon le ratio
        """
        rng = random.Random(random_seed)

        n_distractors = min(round(distractor_ratio * (k_total - 1)), len(distractor_passages))
        n_relevant = k_total - 1 - n_distractors

        selected_distractors = rng.sample(distractor_passages[:50], min(n_distractors, len(distractor_passages)))

        relevant_fillers = [p for p in distractor_passages if p not in selected_distractors][:n_relevant]
        others = relevant_fillers + selected_distractors
        rng.shuffle(others)

        if position == "first":
            assembled = [gold_passage] + others
        elif position == "middle":
            mid = len(others) // 2
            assembled = others[:mid] + [gold_passage] + others[mid:]
        else:
            assembled = others + [gold_passage]

        assembled = assembled[:k_total]
        for rank, p in enumerate(assembled, start=1):
            p.rank = rank

        return assembled

    def passages_to_texts(self, passages: list[RetrievedPassage]) -> list[str]:
        """Extrait les textes des passages."""
        return [p.text for p in passages]

    def log_context_info(self, passages: list[RetrievedPassage], gold_id: str | None = None) -> dict[str, Any]:
        """Journalise les informations sur le contexte assemblé."""
        info: dict[str, Any] = {
            "n_passages": len(passages),
            "passage_ids": [p.passage_id for p in passages],
        }
        if gold_id:
            gold_rank = next((p.rank for p in passages if p.passage_id == gold_id), None)
            info["gold_passage_id"] = gold_id
            info["gold_rank_in_context"] = gold_rank
        return info

"""Métriques RAG : Faithfulness, Context Precision, Answer Relevancy via RAGAS."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class RAGMetrics:
    """
    Calcule les métriques RAG via RAGAS.

    - Faithfulness : proportion d'affirmations soutenues par le contexte
    - Context Precision : proportion de passages utiles dans le top-k
    - Answer Relevancy : adéquation réponse / question

    LLM juge : llama-3-8b (reproductible, pas d'API payante)
    NE PAS utiliser gpt-3.5-turbo en production expérimentale.
    """

    def __init__(self, judge_model: str = "meta-llama/Meta-Llama-3-8B-Instruct") -> None:
        self.judge_model = judge_model
        self._evaluator = None

    def _setup_evaluator(self) -> None:
        """Configure l'évaluateur RAGAS."""
        try:
            from langchain_community.llms import HuggingFacePipeline
            from ragas import evaluate
            from ragas.metrics import answer_relevancy, context_precision, faithfulness
            from transformers import pipeline

            pipe = pipeline(
                "text-generation",
                model=self.judge_model,
                max_new_tokens=512,
                temperature=0.0,
            )
            self._llm = HuggingFacePipeline(pipeline=pipe)
            self._metrics = [faithfulness, context_precision, answer_relevancy]
            self._evaluate_fn = evaluate
            logger.info("RAGAS configuré avec LLM juge : %s", self.judge_model)
        except ImportError as e:
            raise ImportError("ragas et langchain_community requis. pip install ragas") from e

    def compute(
        self,
        questions: list[str],
        answers: list[str],
        contexts_list: list[list[str]],
        ground_truths: list[str] | None = None,
    ) -> dict[str, float]:
        """
        Calcule les métriques RAGAS sur un lot de questions.

        Args:
            questions: liste de questions
            answers: liste de réponses générées
            contexts_list: liste de listes de passages de contexte
            ground_truths: réponses gold (optionnel)

        Returns:
            Dictionnaire avec faithfulness, context_precision, answer_relevancy
        """
        if self._evaluator is None:
            self._setup_evaluator()

        try:
            from datasets import Dataset
        except ImportError as e:
            raise ImportError("datasets requis. pip install datasets") from e

        data: dict[str, Any] = {
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
        }
        if ground_truths:
            data["ground_truth"] = ground_truths

        dataset = Dataset.from_dict(data)
        result = self._evaluate_fn(dataset, metrics=self._metrics, llm=self._llm)

        return {
            "faithfulness": float(result["faithfulness"]),
            "context_precision": float(result["context_precision"]),
            "answer_relevancy": float(result["answer_relevancy"]),
        }

    @staticmethod
    def compute_grounding_error_rate(
        generated_answers: list[str],
        contexts_list: list[list[str]],
        faithfulness_scores: list[float],
        threshold: float = 0.6,
    ) -> dict[str, float]:
        """
        Calcule le taux d'erreurs d'ancrage (affirmations non soutenues).

        Args:
            generated_answers: réponses générées
            contexts_list: contextes correspondants
            faithfulness_scores: scores de fidélité par réponse
            threshold: seuil en dessous duquel une réponse est considérée non ancrée

        Returns:
            Statistiques sur les erreurs d'ancrage
        """
        n = len(faithfulness_scores)
        n_ungrounded = sum(1 for s in faithfulness_scores if s < threshold)
        avg_faith = sum(faithfulness_scores) / n if n > 0 else 0.0

        return {
            "grounding_error_rate": n_ungrounded / n if n > 0 else 0.0,
            "avg_faithfulness": avg_faith,
            "n_ungrounded": n_ungrounded,
            "n_total": n,
            "threshold": threshold,
        }

    @staticmethod
    def compute_abstention_rate(
        predictions: list[str],
        is_unanswerable: list[bool],
        unanswerable_marker: str = "Je ne sais pas",
    ) -> dict[str, float]:
        """
        Calcule le taux d'abstention correcte sur les questions sans réponse (FQuAD 2.0).

        Args:
            predictions: réponses générées
            is_unanswerable: indicateur de question sans réponse
            unanswerable_marker: chaîne indiquant l'abstention

        Returns:
            Statistiques sur l'abstention
        """
        unanswerable_preds = [
            p for p, ua in zip(predictions, is_unanswerable) if ua
        ]
        correct_abstentions = sum(
            1 for p in unanswerable_preds if unanswerable_marker.lower() in p.lower()
        )
        n_unanswerable = len(unanswerable_preds)

        return {
            "abstention_rate": correct_abstentions / n_unanswerable if n_unanswerable > 0 else 0.0,
            "n_unanswerable": n_unanswerable,
            "n_correct_abstentions": correct_abstentions,
        }

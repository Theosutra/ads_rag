"""Métriques QA classiques : Exact Match et F1 token-level."""

from __future__ import annotations

import re
import string
from collections import Counter


class QAMetrics:
    """
    Calcule les métriques QA standard : EM et F1.

    Utilise la bibliothèque HuggingFace evaluate pour la compatibilité
    avec les benchmarks SQuAD/FQuAD.
    """

    def __init__(self) -> None:
        self._em_metric = None
        self._squad_metric = None

    def _load_hf_metrics(self) -> None:
        """Charge les métriques HuggingFace (lazy loading)."""
        import evaluate as hf_evaluate
        if self._em_metric is None:
            self._em_metric = hf_evaluate.load("exact_match")
        if self._squad_metric is None:
            self._squad_metric = hf_evaluate.load("squad")

    @staticmethod
    def normalize_answer(text: str) -> str:
        """Normalise une réponse pour le calcul EM/F1 (lowercase, ponctuation, articles)."""
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\b(a|an|the|le|la|les|un|une|des)\b", " ", text)
        text = " ".join(text.split())
        return text

    def exact_match(self, prediction: str, gold_answers: list[str]) -> int:
        """
        Calcule l'Exact Match entre une prédiction et les réponses gold.

        Returns:
            1 si la prédiction (normalisée) correspond à au moins une réponse gold, 0 sinon
        """
        norm_pred = self.normalize_answer(prediction)
        return int(any(self.normalize_answer(g) == norm_pred for g in gold_answers))

    def token_f1(self, prediction: str, gold_answers: list[str]) -> float:
        """
        Calcule le F1 token-level entre une prédiction et les réponses gold.

        Returns:
            Score F1 maximal sur toutes les réponses gold
        """
        def _f1_single(pred: str, gold: str) -> float:
            pred_tokens = self.normalize_answer(pred).split()
            gold_tokens = self.normalize_answer(gold).split()
            common = Counter(pred_tokens) & Counter(gold_tokens)
            n_common = sum(common.values())
            if n_common == 0:
                return 0.0
            precision = n_common / len(pred_tokens)
            recall = n_common / len(gold_tokens)
            return 2 * precision * recall / (precision + recall)

        return max((_f1_single(prediction, g) for g in gold_answers), default=0.0)

    def compute_batch(
        self,
        predictions: list[str],
        gold_answers_list: list[list[str]],
    ) -> dict[str, float]:
        """
        Calcule EM et F1 moyens sur un lot de prédictions.

        Args:
            predictions: liste des réponses générées
            gold_answers_list: liste des listes de réponses gold

        Returns:
            Dictionnaire avec 'em' et 'f1' moyens
        """
        ems = [self.exact_match(p, g) for p, g in zip(predictions, gold_answers_list)]
        f1s = [self.token_f1(p, g) for p, g in zip(predictions, gold_answers_list)]
        return {
            "em": sum(ems) / len(ems) if ems else 0.0,
            "f1": sum(f1s) / len(f1s) if f1s else 0.0,
            "em_per_sample": ems,
            "f1_per_sample": f1s,
        }

    def compute_squad_format(
        self,
        predictions: list[dict],
        references: list[dict],
    ) -> dict[str, float]:
        """
        Calcule les métriques au format SQuAD via HuggingFace evaluate.

        Args:
            predictions: liste de {"id": ..., "prediction_text": ...}
            references: liste de {"id": ..., "answers": {"text": [...], "answer_start": [...]}}

        Returns:
            Métriques SQuAD (exact_match, f1)
        """
        self._load_hf_metrics()
        return self._squad_metric.compute(predictions=predictions, references=references)

"""Tests statistiques pour la comparaison de systèmes RAG."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class StatisticalTests:
    """
    Tests statistiques pour valider les comparaisons entre configurations RAG.

    Tests disponibles :
    - McNemar : comparaison EM binaire entre deux systèmes appariés
    - Wilcoxon signed-rank : comparaison ΔF1, ΔFaithfulness (non-normal)
    - Permutation test : validation robustesse comparaisons critiques
    - Bootstrap IC 95% : estimation incertitude sur effets moyens
    - Correction Holm-Bonferroni : 18 comparaisons multiples (contrôle FWER)

    Seuils de significativité :
    - *** p < 0.001
    - **  p < 0.01
    - *   p < 0.05
    """

    @staticmethod
    def significance_stars(p_value: float) -> str:
        """Retourne les étoiles de significativité."""
        if p_value < 0.001:
            return "***"
        if p_value < 0.01:
            return "**"
        if p_value < 0.05:
            return "*"
        return "ns"

    @staticmethod
    def mcnemar_test(
        system_a_correct: list[int],
        system_b_correct: list[int],
    ) -> dict[str, Any]:
        """
        Test de McNemar pour la comparaison de deux systèmes sur données binaires (EM).

        Toutes les comparaisons sont appariées par question.

        Args:
            system_a_correct: liste binaire (0/1) indiquant les succès du système A
            system_b_correct: liste binaire (0/1) indiquant les succès du système B

        Returns:
            Statistique de test, p-value et interprétation
        """
        try:
            from mlxtend.evaluate import mcnemar, mcnemar_table
        except ImportError as e:
            raise ImportError("mlxtend requis. pip install mlxtend") from e

        tb = mcnemar_table(
            y_target=np.array(system_a_correct),
            y_model1=np.array(system_a_correct),
            y_model2=np.array(system_b_correct),
        )
        chi2, p = mcnemar(tb, corrected=True)

        return {
            "test": "mcnemar",
            "chi2": float(chi2),
            "p_value": float(p),
            "significant": p < 0.05,
            "stars": StatisticalTests.significance_stars(p),
            "contingency_table": tb.tolist(),
        }

    @staticmethod
    def wilcoxon_test(
        scores_a: list[float],
        scores_b: list[float],
        alternative: str = "two-sided",
    ) -> dict[str, Any]:
        """
        Test de Wilcoxon signed-rank pour la comparaison de scores continus (F1, Faithfulness).

        Args:
            scores_a: scores du système A
            scores_b: scores du système B
            alternative: 'two-sided', 'greater', 'less'

        Returns:
            Statistique de test, p-value et interprétation
        """
        diffs = np.array(scores_a) - np.array(scores_b)

        if np.all(diffs == 0):
            return {
                "test": "wilcoxon",
                "statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
                "stars": "ns",
                "mean_diff": 0.0,
            }

        from scipy import stats
        stat, p = stats.wilcoxon(scores_a, scores_b, alternative=alternative)

        return {
            "test": "wilcoxon",
            "statistic": float(stat),
            "p_value": float(p),
            "significant": p < 0.05,
            "stars": StatisticalTests.significance_stars(p),
            "mean_diff": float(np.mean(diffs)),
            "median_diff": float(np.median(diffs)),
        }

    @staticmethod
    def permutation_test(
        scores_a: list[float],
        scores_b: list[float],
        n_permutations: int = 10000,
        random_seed: int = 42,
    ) -> dict[str, Any]:
        """
        Test de permutation pour la validation robuste des comparaisons critiques.

        Args:
            scores_a: scores du système A
            scores_b: scores du système B
            n_permutations: nombre de permutations
            random_seed: graine pour la reproductibilité

        Returns:
            Statistique observée, p-value et distribution nulle
        """
        rng = np.random.default_rng(random_seed)
        a = np.array(scores_a)
        b = np.array(scores_b)

        observed_diff = np.mean(a) - np.mean(b)

        combined = np.concatenate([a, b])
        n = len(a)

        null_diffs = []
        for _ in range(n_permutations):
            perm = rng.permutation(combined)
            null_diff = np.mean(perm[:n]) - np.mean(perm[n:])
            null_diffs.append(null_diff)

        null_diffs = np.array(null_diffs)
        p_value = np.mean(np.abs(null_diffs) >= np.abs(observed_diff))

        return {
            "test": "permutation",
            "observed_diff": float(observed_diff),
            "p_value": float(p_value),
            "n_permutations": n_permutations,
            "significant": p_value < 0.05,
            "stars": StatisticalTests.significance_stars(p_value),
        }

    @staticmethod
    def bootstrap_ci(
        scores: list[float],
        n_bootstrap: int = 10000,
        confidence: float = 0.95,
        random_seed: int = 42,
    ) -> dict[str, Any]:
        """
        Estime l'intervalle de confiance par bootstrap (IC 95%).

        Args:
            scores: scores à analyser
            n_bootstrap: nombre de répétitions bootstrap
            confidence: niveau de confiance (défaut : 0.95)
            random_seed: graine pour la reproductibilité

        Returns:
            Moyenne, IC inférieur et supérieur
        """
        rng = np.random.default_rng(random_seed)
        arr = np.array(scores)
        means = []
        for _ in range(n_bootstrap):
            sample = rng.choice(arr, size=len(arr), replace=True)
            means.append(np.mean(sample))

        alpha = 1 - confidence
        ci_lower = float(np.percentile(means, 100 * alpha / 2))
        ci_upper = float(np.percentile(means, 100 * (1 - alpha / 2)))

        return {
            "mean": float(np.mean(arr)),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "confidence": confidence,
            "n_bootstrap": n_bootstrap,
        }

    @staticmethod
    def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> dict[str, Any]:
        """
        Correction Holm-Bonferroni pour comparaisons multiples (contrôle FWER).

        Utilisée pour les 18 comparaisons multiples du protocole A.

        Args:
            p_values: liste des p-values des tests individuels
            alpha: niveau de significativité global (défaut : 0.05)

        Returns:
            P-values corrigées et décisions de rejet
        """
        n = len(p_values)
        indexed = sorted(enumerate(p_values), key=lambda x: x[1])

        adjusted = [None] * n
        rejected = [False] * n

        for rank, (orig_idx, p) in enumerate(indexed):
            threshold = alpha / (n - rank)
            adjusted[orig_idx] = min(p * (n - rank), 1.0)
            if p <= threshold:
                rejected[orig_idx] = True

        for i in range(1, n):
            orig_idx, _ = indexed[i]
            prev_idx, _ = indexed[i - 1]
            if rejected[prev_idx] is False:
                rejected[orig_idx] = False

        return {
            "original_p_values": p_values,
            "adjusted_p_values": adjusted,
            "rejected": rejected,
            "n_comparisons": n,
            "alpha": alpha,
            "n_significant": sum(rejected),
        }

    def compare_systems(
        self,
        system_a_name: str,
        system_b_name: str,
        em_a: list[int],
        em_b: list[int],
        f1_a: list[float],
        f1_b: list[float],
        faithfulness_a: list[float] | None = None,
        faithfulness_b: list[float] | None = None,
    ) -> dict[str, Any]:
        """
        Effectue une batterie complète de tests statistiques entre deux systèmes.

        Args:
            system_a_name: nom du système A
            system_b_name: nom du système B
            em_a, em_b: scores EM par question (0/1)
            f1_a, f1_b: scores F1 par question
            faithfulness_a, faithfulness_b: scores Faithfulness (optionnel)

        Returns:
            Résultats de tous les tests
        """
        results: dict[str, Any] = {
            "system_a": system_a_name,
            "system_b": system_b_name,
            "n_samples": len(em_a),
        }

        results["mcnemar_em"] = self.mcnemar_test(em_a, em_b)
        results["wilcoxon_f1"] = self.wilcoxon_test(f1_a, f1_b)
        results["permutation_f1"] = self.permutation_test(f1_a, f1_b)
        results["bootstrap_f1_a"] = self.bootstrap_ci(f1_a)
        results["bootstrap_f1_b"] = self.bootstrap_ci(f1_b)

        if faithfulness_a and faithfulness_b:
            results["wilcoxon_faithfulness"] = self.wilcoxon_test(faithfulness_a, faithfulness_b)
            results["bootstrap_faithfulness_a"] = self.bootstrap_ci(faithfulness_a)
            results["bootstrap_faithfulness_b"] = self.bootstrap_ci(faithfulness_b)

        return results

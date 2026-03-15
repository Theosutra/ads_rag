"""Figures pour le mémoire : heatmaps, courbes dose-réponse, distributions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

FIGURE_DIR = Path("reports/figures/")
STYLE = "seaborn-v0_8-whitegrid"


class ExperimentVisualizer:
    """Génère toutes les figures du mémoire."""

    def __init__(self, output_dir: str = "reports/figures/", dpi: int = 150) -> None:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        import numpy as np
        import pandas as pd

        self._plt = plt
        self._mticker = mticker
        self._np = np
        self._pd = pd

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        try:
            plt.style.use(STYLE)
        except OSError:
            plt.style.use("default")

    def _save(self, fig: Any, filename: str) -> str:
        path = self.output_dir / filename
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        self._plt.close(fig)
        logger.info("Figure sauvegardée → %s", path)
        return str(path)

    def plot_retriever_comparison(
        self,
        df: Any,
        metric: str = "f1",
        output_file: str = "fig_retriever_comparison.png",
    ) -> str:
        """
        Barplot comparant les performances des différents retrievers (Protocole A).

        Illustre H1 : saturation à k=10 et H4 : variabilité FR vs EN.
        """
        plt = self._plt
        mticker = self._mticker

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        for ax, lang in zip(axes, ["fr", "en"]):
            sub = df[df["lang"] == lang].copy()
            pivot = sub.pivot_table(index="retriever", columns="k", values=metric, aggfunc="mean")
            pivot.plot(kind="bar", ax=ax, colormap="viridis", edgecolor="white")
            ax.set_title(f"{metric.upper()} — {lang.upper()}", fontsize=13)
            ax.set_xlabel("Retriever")
            ax.set_ylabel(metric.upper())
            ax.legend(title="k", bbox_to_anchor=(1, 1))
            ax.tick_params(axis="x", rotation=30)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

        fig.suptitle("Comparaison des retrievers par langue", fontsize=14, fontweight="bold")
        fig.tight_layout()
        return self._save(fig, output_file)

    def plot_k_saturation(
        self,
        df: Any,
        metric: str = "f1",
        output_file: str = "fig_k_saturation.png",
    ) -> str:
        """
        Courbe de saturation : performance en fonction de k (H1).

        Montre le point de saturation attendu autour de k=10.
        """
        plt = self._plt

        fig, ax = plt.subplots(figsize=(8, 5))

        for retriever in df["retriever"].unique():
            sub = df[df["retriever"] == retriever].groupby("k")[metric].mean().reset_index()
            ax.plot(sub["k"], sub[metric], marker="o", label=retriever, linewidth=2)

        ax.axvline(x=10, color="red", linestyle="--", alpha=0.6, label="k=10 (saturation attendue)")
        ax.set_xlabel("Nombre de passages k", fontsize=12)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_title(f"Saturation des performances en fonction de k — {metric.upper()}", fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.5)
        fig.tight_layout()
        return self._save(fig, output_file)

    def plot_distractor_dose_response(
        self,
        results: list[dict[str, Any]],
        metric: str = "faithfulness",
        output_file: str = "fig_distractor_dose_response.png",
    ) -> str:
        """
        Courbe dose-réponse : effet des distracteurs sur la factualité (Protocole B, H1).

        Montre le seuil critique attendu à p=0.4.
        """
        plt = self._plt
        np = self._np

        ratios = sorted(set(r["distractor_ratio"] for r in results))
        means = []
        stds = []

        for ratio in ratios:
            scores = [r[metric] for r in results if r["distractor_ratio"] == ratio]
            means.append(np.mean(scores))
            stds.append(np.std(scores))

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(ratios, means, yerr=stds, marker="o", linewidth=2, capsize=4, color="steelblue")
        ax.axvline(x=0.4, color="orange", linestyle="--", alpha=0.8, label="Seuil critique p=0.4")
        ax.axvline(x=0.6, color="red", linestyle="--", alpha=0.8, label="Dégradation marquée p=0.6")
        ax.set_xlabel("Proportion de distracteurs p", fontsize=12)
        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.set_title("Effet des distracteurs sur la factualité (Protocole B)", fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.5)
        fig.tight_layout()
        return self._save(fig, output_file)

    def plot_position_effect(
        self,
        results: list[dict[str, Any]],
        metric: str = "faithfulness",
        output_file: str = "fig_position_effect.png",
    ) -> str:
        """
        Barplot de l'effet de position du passage gold (Lost-in-the-Middle, H3).
        """
        plt = self._plt
        np = self._np

        positions = ["first", "middle", "last"]
        scores_by_pos = {pos: [] for pos in positions}

        for r in results:
            pos = r.get("position", "first")
            if pos in scores_by_pos:
                scores_by_pos[pos].append(r.get(metric, 0.0))

        means = [np.mean(scores_by_pos[p]) if scores_by_pos[p] else 0.0 for p in positions]
        stds = [np.std(scores_by_pos[p]) if scores_by_pos[p] else 0.0 for p in positions]

        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(positions, means, yerr=stds, capsize=5,
                      color=["#2196F3", "#FF9800", "#F44336"], edgecolor="white")
        ax.set_xlabel("Position du passage gold", fontsize=12)
        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.set_title("Effet de position du passage gold (Lost-in-the-Middle)", fontsize=13)
        ax.set_ylim(0, 1.1)

        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{mean:.3f}", ha="center", fontsize=10)

        fig.tight_layout()
        return self._save(fig, output_file)

    def plot_reranking_effect(
        self,
        df: Any,
        output_file: str = "fig_reranking_effect.png",
    ) -> str:
        """
        Comparaison avec/sans reranking pour tester H2 :
        le reranking améliore Faithfulness plutôt que EM/F1.
        """
        plt = self._plt
        np = self._np

        metrics = ["em", "f1", "faithfulness"]
        labels = ["Exact Match", "F1", "Faithfulness"]
        colors_no = "#90CAF9"
        colors_yes = "#1565C0"

        df_no = df[df["reranking"] == False].groupby("retriever")[metrics].mean()
        df_yes = df[df["reranking"] == True].groupby("retriever")[metrics].mean()

        retrievers = df["retriever"].unique()
        x = np.arange(len(metrics))
        width = 0.35

        fig, axes = plt.subplots(1, len(retrievers), figsize=(5 * len(retrievers), 5), sharey=True)
        if len(retrievers) == 1:
            axes = [axes]

        for ax, retriever in zip(axes, retrievers):
            if retriever in df_no.index and retriever in df_yes.index:
                vals_no = [df_no.loc[retriever, m] for m in metrics]
                vals_yes = [df_yes.loc[retriever, m] for m in metrics]
                ax.bar(x - width / 2, vals_no, width, label="Sans reranking", color=colors_no, edgecolor="white")
                ax.bar(x + width / 2, vals_yes, width, label="Avec reranking", color=colors_yes, edgecolor="white")

            ax.set_title(f"Retriever : {retriever}", fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylim(0, 1.05)
            ax.legend()

        fig.suptitle("Impact du reranking par métrique (H2)", fontsize=14, fontweight="bold")
        fig.tight_layout()
        return self._save(fig, output_file)

    def plot_error_distribution(
        self,
        distribution: dict[str, Any],
        output_file: str = "fig_error_distribution.png",
    ) -> str:
        """
        Camembert de la distribution des types d'erreurs (taxonomie).
        """
        plt = self._plt

        dist = distribution.get("distribution", {})
        labels = list(dist.keys())
        sizes = [dist[k]["count"] for k in labels]

        fig, ax = plt.subplots(figsize=(9, 7))
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            pctdistance=0.8,
        )
        for text in autotexts:
            text.set_fontsize(9)
        ax.set_title("Distribution des types d'erreurs RAG\n(annotation manuelle — 120 cas)", fontsize=13)
        fig.tight_layout()
        return self._save(fig, output_file)

    def plot_metrics_heatmap(
        self,
        df: Any,
        metric: str = "faithfulness",
        output_file: str = "fig_heatmap_configs.png",
    ) -> str:
        """
        Heatmap des métriques par configuration (retriever × format_contexte).
        """
        plt = self._plt
        np = self._np

        pivot = df.pivot_table(
            index="retriever",
            columns="format",
            values=metric,
            aggfunc="mean",
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)

        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels(pivot.columns, fontsize=11)
        ax.set_yticklabels(pivot.index, fontsize=11)
        ax.set_title(f"Heatmap — {metric.capitalize()} par configuration", fontsize=13)

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            color="black" if val < 0.7 else "white", fontsize=10)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        return self._save(fig, output_file)

    def generate_all_figures(self, df: Any) -> list[str]:
        """Génère toutes les figures du mémoire à partir d'un DataFrame de résultats."""
        saved = []

        if "retriever" in df.columns and "lang" in df.columns and "k" in df.columns:
            saved.append(self.plot_retriever_comparison(df))
            saved.append(self.plot_k_saturation(df))

        if "reranking" in df.columns:
            saved.append(self.plot_reranking_effect(df))

        if "retriever" in df.columns and "format" in df.columns and "faithfulness" in df.columns:
            saved.append(self.plot_metrics_heatmap(df))

        return saved

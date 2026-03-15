"""Protocole A — Ablation end-to-end bilingue."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from src.eval.qa_metrics import QAMetrics
from src.eval.retrieval_metrics import RetrievalMetrics
from src.ingestion.loader import DatasetLoader
from src.rag.context_builder import ContextBuilder
from src.rag.generator import RAGGenerator
from src.rag.prompt_formatter import PromptFormatter
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import get_reranker

from .base_experiment import BaseExperiment

logger = logging.getLogger(__name__)

FACTORIAL_PLAN = [
    {"retriever": "bm25",   "k": 5,  "reranking": False, "format": "standard"},
    {"retriever": "bm25",   "k": 10, "reranking": False, "format": "standard"},
    {"retriever": "bm25",   "k": 20, "reranking": False, "format": "standard"},
    {"retriever": "dense",  "k": 5,  "reranking": False, "format": "standard"},
    {"retriever": "dense",  "k": 10, "reranking": False, "format": "standard"},
    {"retriever": "dense",  "k": 20, "reranking": False, "format": "standard"},
    {"retriever": "hybrid", "k": 5,  "reranking": False, "format": "standard"},
    {"retriever": "hybrid", "k": 10, "reranking": False, "format": "standard"},
    {"retriever": "hybrid", "k": 20, "reranking": False, "format": "standard"},
    {"retriever": "bm25",   "k": 10, "reranking": True,  "format": "standard", "initial_k": 50},
    {"retriever": "dense",  "k": 10, "reranking": True,  "format": "standard", "initial_k": 50},
    {"retriever": "bm25",   "k": 10, "reranking": False, "format": "citations"},
    {"retriever": "dense",  "k": 10, "reranking": False, "format": "citations"},
    {"retriever": "hybrid", "k": 10, "reranking": False, "format": "citations"},
    {"retriever": "hybrid", "k": 10, "reranking": True,  "format": "citations", "initial_k": 50},
]


class ProtocolA(BaseExperiment):
    """
    Protocole A — Plan factoriel complet bilingue.

    Teste : H1 (saturation à k=10) et H2 (reranking → faithfulness).

    Design :
        retriever × k × reranking × format_contexte × langue + baseline
    """

    def __init__(
        self,
        output_dir: str = "runs/A_ablation/",
        config_dir: str = "configs/",
        seed: int = 42,
    ) -> None:
        super().__init__(output_dir, config_dir, seed)
        self.qa_metrics = QAMetrics()
        self.retrieval_metrics = RetrievalMetrics()
        self.formatter = PromptFormatter()
        self.context_builder = ContextBuilder()
        self.generator = RAGGenerator()

    def _get_retriever(self, retriever_name: str, lang: str):
        """Instancie le retriever selon le nom."""
        cfg = self._retriever_config

        if retriever_name == "bm25":
            return BM25Retriever(
                index_dir=cfg["bm25"]["index_dir"],
                k1=cfg["bm25"]["k1"],
                b=cfg["bm25"]["b"],
            )
        if retriever_name == "dense":
            return DenseRetriever(
                index_dir=cfg["dense"]["index_dir"],
                lang=lang,
            )
        if retriever_name == "hybrid":
            bm25 = BM25Retriever(index_dir=cfg["bm25"]["index_dir"])
            dense = DenseRetriever(index_dir=cfg["dense"]["index_dir"], lang=lang)
            return HybridRetriever(bm25, dense, alpha=cfg["hybrid"]["alpha"])

        raise ValueError(f"Retriever inconnu : {retriever_name}")

    def run_baseline(self, samples: list[dict], lang: str) -> None:
        """Exécute la configuration baseline (sans retrieval)."""
        run_file = self.get_run_file("baseline", lang)
        config = {"retriever": None, "k": 0, "reranking": False, "format": "standard", "lang": lang, "protocol": "A"}

        for sample in tqdm(samples, desc="Baseline"):
            prompt = self.formatter.format_baseline(sample["question"])
            answer = self.generator.generate(prompt)
            gold = sample["answers"]
            gold_list = [gold] if isinstance(gold, str) else gold

            metrics = {
                "em": self.qa_metrics.exact_match(answer, gold_list),
                "f1": self.qa_metrics.token_f1(answer, gold_list),
            }
            self.log_result(
                run_file=run_file,
                question_id=sample["id"],
                question=sample["question"],
                gold_answer=gold_list,
                config=config,
                retrieved_passages=[],
                prompt=prompt,
                generated_answer=answer,
                metrics=metrics,
            )

    def run_config(self, samples: list[dict], exp_config: dict, lang: str) -> None:
        """Exécute une configuration expérimentale du plan factoriel."""
        config_name = (
            f"{exp_config['retriever']}_k{exp_config['k']}"
            f"{'_rerank' if exp_config.get('reranking') else ''}"
            f"_{exp_config['format']}"
        )
        run_file = self.get_run_file(config_name, lang)
        full_config = {**exp_config, "lang": lang, "protocol": "A", "generator": "google/mt5-large"}

        retriever = self._get_retriever(exp_config["retriever"], lang)
        reranker = get_reranker("colbert") if exp_config.get("reranking") else None

        for sample in tqdm(samples, desc=config_name):
            initial_k = exp_config.get("initial_k", exp_config["k"])
            passages = retriever.retrieve(sample["question"], k=initial_k)

            if reranker and passages:
                passages = reranker.rerank(sample["question"], passages, top_k=exp_config["k"])

            passages = passages[:exp_config["k"]]
            passage_texts = [p.text for p in passages]

            prompt = self.formatter.format(sample["question"], passage_texts, fmt=exp_config["format"])
            answer = self.generator.generate(prompt)

            gold = sample["answers"]
            gold_list = [gold] if isinstance(gold, str) else gold

            metrics: dict[str, Any] = {
                "em": self.qa_metrics.exact_match(answer, gold_list),
                "f1": self.qa_metrics.token_f1(answer, gold_list),
            }

            self.log_result(
                run_file=run_file,
                question_id=sample["id"],
                question=sample["question"],
                gold_answer=gold_list,
                config=full_config,
                retrieved_passages=[p.to_dict() for p in passages],
                prompt=prompt,
                generated_answer=answer,
                metrics=metrics,
            )

    def run(self, lang: str = "fr", max_samples: int | None = None) -> None:
        """
        Exécute le protocole A complet pour une langue.

        Args:
            lang: 'fr' ou 'en'
            max_samples: limiter le nombre d'exemples (utile pour les tests)
        """
        logger.info("=== Protocole A — Langue : %s ===", lang.upper())

        loader = DatasetLoader()
        if lang == "fr":
            samples = loader.load_fquad2("validation")
        else:
            samples = loader.load_kilt_nq("validation")

        if max_samples:
            samples = samples[:max_samples]

        logger.info("%d questions chargées", len(samples))

        self.run_baseline(samples, lang)

        for exp_config in FACTORIAL_PLAN:
            self.run_config(samples, exp_config, lang)

        logger.info("=== Protocole A terminé — %s ===", lang.upper())

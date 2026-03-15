"""Protocole C — Factualité fine et attribution."""

from __future__ import annotations

import logging
from typing import Any

from tqdm import tqdm

from src.eval.qa_metrics import QAMetrics
from src.eval.rag_metrics import RAGMetrics
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

PROTOCOL_C_CONFIGS = [
    {
        "name": "RAG_standard",
        "retriever": "hybrid",
        "k": 10,
        "reranking": False,
        "format": "standard",
        "description": "Pipeline RAG simple, format concaténation",
    },
    {
        "name": "RAG_rerank",
        "retriever": "hybrid",
        "k": 10,
        "reranking": True,
        "initial_k": 50,
        "format": "standard",
        "description": "RAG + étape reranking ColBERT",
    },
    {
        "name": "RAG_citations",
        "retriever": "hybrid",
        "k": 10,
        "reranking": False,
        "format": "citations",
        "description": "Format citations strictes + instruction abstention",
    },
    {
        "name": "RAG_verify",
        "retriever": "hybrid",
        "k": 10,
        "reranking": True,
        "initial_k": 50,
        "format": "citations",
        "description": "Inspiration Self-RAG : vérification cohérence post-génération",
    },
]


class ProtocolC(BaseExperiment):
    """
    Protocole C — Factualité fine et attribution.

    Teste H2 et H3 sur les hallucinations résiduelles.

    4 configurations :
    - RAG_standard : pipeline simple concaténation
    - RAG_rerank : avec reranking ColBERT
    - RAG_citations : format citations strictes
    - RAG_verify : inspiration Self-RAG
    """

    def __init__(
        self,
        output_dir: str = "runs/C_factuality/",
        config_dir: str = "configs/",
        seed: int = 42,
    ) -> None:
        super().__init__(output_dir, config_dir, seed)
        self.qa_metrics = QAMetrics()
        self.rag_metrics = RAGMetrics()
        self.formatter = PromptFormatter()
        self.context_builder = ContextBuilder()
        self.generator = RAGGenerator()

    def _build_retriever(self, lang: str):
        """Construit le retriever hybride."""
        cfg = self._retriever_config
        bm25 = BM25Retriever(index_dir=cfg["bm25"]["index_dir"])
        dense = DenseRetriever(index_dir=cfg["dense"]["index_dir"], lang=lang)
        return HybridRetriever(bm25, dense, alpha=cfg["hybrid"]["alpha"])

    def _verify_answer(self, question: str, answer: str, context_passages: list[str]) -> tuple[str, float]:
        """
        Vérification post-génération inspirée de Self-RAG.

        Vérifie si la réponse est supportée par le contexte.
        Retourne la réponse vérifiée et un score de confiance.
        """
        if not answer or "je ne sais pas" in answer.lower():
            return answer, 1.0

        context = " ".join(context_passages)
        key_terms = set(answer.lower().split())

        supported_terms = sum(1 for t in key_terms if len(t) > 3 and t in context.lower())
        confidence = supported_terms / max(len([t for t in key_terms if len(t) > 3]), 1)

        if confidence < 0.3:
            return "Je ne sais pas", confidence

        return answer, confidence

    def run_config(
        self,
        samples: list[dict],
        exp_config: dict,
        lang: str,
        max_samples: int | None = None,
    ) -> None:
        """Exécute une configuration du Protocole C."""
        if max_samples:
            samples = samples[:max_samples]

        config_name = exp_config["name"]
        run_file = self.get_run_file(config_name, lang)
        retriever = self._build_retriever(lang)
        reranker = get_reranker("colbert") if exp_config.get("reranking") else None

        full_config: dict[str, Any] = {
            **exp_config,
            "lang": lang,
            "protocol": "C",
            "generator": "google/mt5-large",
        }

        all_questions, all_answers, all_contexts = [], [], []

        for sample in tqdm(samples, desc=f"Protocole C — {config_name}"):
            initial_k = exp_config.get("initial_k", exp_config["k"])
            passages = retriever.retrieve(sample["question"], k=initial_k)

            if reranker and passages:
                passages = reranker.rerank(sample["question"], passages, top_k=exp_config["k"])

            passages = passages[:exp_config["k"]]
            passage_texts = [p.text for p in passages]

            prompt = self.formatter.format(sample["question"], passage_texts, fmt=exp_config["format"])
            answer = self.generator.generate(prompt)

            if config_name == "RAG_verify":
                answer, confidence = self._verify_answer(sample["question"], answer, passage_texts)
            else:
                confidence = None

            gold = sample.get("answers", [])
            gold_list = [gold] if isinstance(gold, str) else gold
            is_unanswerable = sample.get("is_impossible", False)

            metrics: dict[str, Any] = {
                "em": self.qa_metrics.exact_match(answer, gold_list),
                "f1": self.qa_metrics.token_f1(answer, gold_list),
            }
            if confidence is not None:
                metrics["self_rag_confidence"] = confidence

            all_questions.append(sample["question"])
            all_answers.append(answer)
            all_contexts.append(passage_texts)

            extra: dict[str, Any] = {"is_unanswerable": is_unanswerable}

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
                extra=extra,
            )

        logger.info("Config %s terminée — %d exemples traités", config_name, len(all_questions))

    def run(self, lang: str = "fr", max_samples: int | None = None) -> None:
        """Exécute le protocole C complet."""
        logger.info("=== Protocole C — Langue : %s ===", lang.upper())

        loader = DatasetLoader()
        if lang == "fr":
            samples = loader.load_fquad2("validation")
        else:
            samples = loader.load_kilt_nq("validation")

        for exp_config in PROTOCOL_C_CONFIGS:
            self.run_config(samples, exp_config, lang, max_samples)

        logger.info("=== Protocole C terminé ===")

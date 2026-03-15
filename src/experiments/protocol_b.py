"""Protocole B — Dégradation contrôlée du contexte."""

from __future__ import annotations

import logging
from typing import Any

from tqdm import tqdm

from src.eval.qa_metrics import QAMetrics
from src.ingestion.loader import DatasetLoader
from src.rag.context_builder import ContextBuilder
from src.rag.generator import RAGGenerator
from src.rag.prompt_formatter import PromptFormatter
from src.retrieval.bm25_retriever import BM25Retriever, RetrievedPassage

from .base_experiment import BaseExperiment

logger = logging.getLogger(__name__)

DISTRACTOR_RATIOS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
POSITIONS = ["first", "middle", "last"]
K_TOTAL = 10


class ProtocolB(BaseExperiment):
    """
    Protocole B — Dégradation contrôlée du contexte.

    Teste H1 (seuil de bruit) et H3 (effet de position).

    Design dose-réponse :
    - p ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0} proportion de distracteurs
    - position ∈ {first, middle, last} du passage gold
    - Deux types de distracteurs : aléatoires et semi-pertinents
    """

    def __init__(
        self,
        output_dir: str = "runs/B_context_noise/",
        config_dir: str = "configs/",
        seed: int = 42,
    ) -> None:
        super().__init__(output_dir, config_dir, seed)
        self.qa_metrics = QAMetrics()
        self.formatter = PromptFormatter()
        self.context_builder = ContextBuilder()
        self.generator = RAGGenerator()

    def _get_gold_passage(self, sample: dict, retriever: BM25Retriever) -> RetrievedPassage | None:
        """Récupère le passage gold via l'identifiant de provenance KILT/FQuAD."""
        provenance = sample.get("provenance", [])
        if not provenance:
            return None

        gold_id = str(provenance[0].get("wikipedia_id", ""))
        passages = retriever.retrieve(sample["question"], k=100)
        for p in passages:
            if gold_id in p.passage_id:
                return p

        if passages:
            return passages[0]
        return None

    def _get_random_distractors(
        self,
        corpus_sample: list[RetrievedPassage],
        gold_id: str,
        n: int,
        seed: int,
    ) -> list[RetrievedPassage]:
        """Sélectionne des distracteurs aléatoires."""
        import random
        rng = random.Random(seed)
        candidates = [p for p in corpus_sample if p.passage_id != gold_id]
        return rng.sample(candidates, min(n, len(candidates)))

    def _get_semi_relevant_distractors(
        self,
        question: str,
        retriever: BM25Retriever,
        gold_id: str,
        n: int,
    ) -> list[RetrievedPassage]:
        """Sélectionne des distracteurs semi-pertinents (top-k sans le passage gold)."""
        candidates = retriever.retrieve(question, k=50 + n)
        return [p for p in candidates if p.passage_id != gold_id][:n]

    def run_dose_response(
        self,
        samples: list[dict],
        retriever: BM25Retriever,
        distractor_type: str = "random",
        lang: str = "fr",
        max_samples: int | None = None,
    ) -> None:
        """
        Exécute l'étude dose-réponse sur la proportion de distracteurs.

        Args:
            samples: questions avec annotations de provenance
            retriever: retriever utilisé pour les distracteurs semi-pertinents
            distractor_type: 'random' ou 'semi_relevant'
            lang: langue de l'expérience
            max_samples: limite d'exemples
        """
        if max_samples:
            samples = samples[:max_samples]

        corpus_sample = retriever.retrieve("Paris capitale France", k=200)

        for ratio in DISTRACTOR_RATIOS:
            config_name = f"dose_response_p{str(ratio).replace('.', '')}_dist{distractor_type}"
            run_file = self.get_run_file(config_name, lang)
            config: dict[str, Any] = {
                "protocol": "B",
                "distractor_ratio": ratio,
                "distractor_type": distractor_type,
                "k_total": K_TOTAL,
                "lang": lang,
                "generator": "google/mt5-large",
            }

            for sample in tqdm(samples, desc=f"Protocole B — p={ratio}"):
                gold_passage = self._get_gold_passage(sample, retriever)
                if gold_passage is None:
                    continue

                n_dist = round(ratio * (K_TOTAL - 1))

                if distractor_type == "random":
                    distractors = self._get_random_distractors(
                        corpus_sample, gold_passage.passage_id, n_dist + 20, self.seed
                    )
                else:
                    distractors = self._get_semi_relevant_distractors(
                        sample["question"], retriever, gold_passage.passage_id, n_dist + 20
                    )

                context_passages = self.context_builder.build_with_distractors(
                    gold_passage=gold_passage,
                    distractor_passages=distractors,
                    k_total=K_TOTAL,
                    distractor_ratio=ratio,
                    position="first",
                    random_seed=self.seed,
                )

                passage_texts = [p.text for p in context_passages]
                prompt = self.formatter.format_standard(sample["question"], passage_texts)
                answer = self.generator.generate(prompt)

                gold = sample.get("answers", [])
                gold_list = [gold] if isinstance(gold, str) else gold

                metrics: dict[str, Any] = {
                    "em": self.qa_metrics.exact_match(answer, gold_list),
                    "f1": self.qa_metrics.token_f1(answer, gold_list),
                    "distractor_ratio": ratio,
                }

                self.log_result(
                    run_file=run_file,
                    question_id=sample["id"],
                    question=sample["question"],
                    gold_answer=gold_list,
                    config=config,
                    retrieved_passages=[p.to_dict() for p in context_passages],
                    prompt=prompt,
                    generated_answer=answer,
                    metrics=metrics,
                )

    def run_position_study(
        self,
        samples: list[dict],
        retriever: BM25Retriever,
        lang: str = "fr",
        max_samples: int | None = None,
    ) -> None:
        """
        Étudie l'effet de position du passage gold (Lost-in-the-Middle).

        Teste H3 en plaçant le passage gold en first / middle / last.
        """
        if max_samples:
            samples = samples[:max_samples]

        corpus_sample = retriever.retrieve("Wikipedia article", k=200)

        for position in POSITIONS:
            config_name = f"position_{position}"
            run_file = self.get_run_file(config_name, lang)
            config: dict[str, Any] = {
                "protocol": "B",
                "position": position,
                "k_total": K_TOTAL,
                "distractor_ratio": 0.4,
                "lang": lang,
                "generator": "google/mt5-large",
            }

            for sample in tqdm(samples, desc=f"Position — {position}"):
                gold_passage = self._get_gold_passage(sample, retriever)
                if gold_passage is None:
                    continue

                distractors = self._get_random_distractors(
                    corpus_sample, gold_passage.passage_id, K_TOTAL + 10, self.seed
                )

                context_passages = self.context_builder.build_with_distractors(
                    gold_passage=gold_passage,
                    distractor_passages=distractors,
                    k_total=K_TOTAL,
                    distractor_ratio=0.4,
                    position=position,
                    random_seed=self.seed,
                )

                passage_texts = [p.text for p in context_passages]
                prompt = self.formatter.format_standard(sample["question"], passage_texts)
                answer = self.generator.generate(prompt)

                gold = sample.get("answers", [])
                gold_list = [gold] if isinstance(gold, str) else gold

                metrics: dict[str, Any] = {
                    "em": self.qa_metrics.exact_match(answer, gold_list),
                    "f1": self.qa_metrics.token_f1(answer, gold_list),
                    "position": position,
                }

                self.log_result(
                    run_file=run_file,
                    question_id=sample["id"],
                    question=sample["question"],
                    gold_answer=gold_list,
                    config=config,
                    retrieved_passages=[p.to_dict() for p in context_passages],
                    prompt=prompt,
                    generated_answer=answer,
                    metrics=metrics,
                )

    def run(self, lang: str = "fr", max_samples: int | None = None) -> None:
        """Exécute le protocole B complet."""
        logger.info("=== Protocole B — Langue : %s ===", lang.upper())

        loader = DatasetLoader()
        if lang == "fr":
            samples = loader.load_fquad2("validation")
        else:
            samples = loader.load_kilt_nq("validation")

        retriever = BM25Retriever(
            index_dir=self._retriever_config["bm25"]["index_dir"],
            k1=self._retriever_config["bm25"]["k1"],
            b=self._retriever_config["bm25"]["b"],
        )

        self.run_dose_response(samples, retriever, "random", lang, max_samples)
        self.run_dose_response(samples, retriever, "semi_relevant", lang, max_samples)
        self.run_position_study(samples, retriever, lang, max_samples)

        logger.info("=== Protocole B terminé ===")

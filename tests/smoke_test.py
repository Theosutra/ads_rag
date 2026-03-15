"""
Smoke test — Vérifie que tous les modules s'importent et fonctionnent correctement.

Exécution : python -m tests.smoke_test --fast
Durée cible : < 5 minutes
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

PASS = "✓"
FAIL = "✗"


def test_imports() -> bool:
    """Vérifie que tous les modules Python s'importent sans erreur."""
    logger.info("Test 1 : Imports des modules...")
    errors = []
    warnings = []

    HEAVY_DEPS = {"torch", "transformers", "jsonlines", "faiss"}

    modules = [
        "src.ingestion.loader",
        "src.ingestion.normalizer",
        "src.ingestion.chunker",
        "src.retrieval.bm25_retriever",
        "src.retrieval.dense_retriever",
        "src.retrieval.hybrid_retriever",
        "src.retrieval.reranker",
        "src.rag.context_builder",
        "src.rag.prompt_formatter",
        "src.rag.generator",
        "src.eval.qa_metrics",
        "src.eval.retrieval_metrics",
        "src.eval.rag_metrics",
        "src.eval.statistical_tests",
        "src.analysis.error_taxonomy",
        "src.analysis.visualizations",
    ]

    for module_name in modules:
        try:
            __import__(module_name)
            logger.info("  %s %s", PASS, module_name)
        except ImportError as e:
            missing = str(e).replace("No module named ", "").strip("'")
            if any(dep in missing for dep in HEAVY_DEPS):
                logger.warning("  ~ %s (dép. lourde manquante : %s)", module_name, missing)
                warnings.append(module_name)
            else:
                logger.error("  %s %s : %s", FAIL, module_name, e)
                errors.append(module_name)
        except Exception as e:
            logger.error("  %s %s : %s", FAIL, module_name, e)
            errors.append(module_name)

    if warnings:
        logger.warning("Modules avec dépendances lourdes non installées : %s", warnings)
        logger.warning("Installer avec : pip install -r requirements.txt")
    if errors:
        logger.error("Échecs réels : %s", errors)
        return False

    logger.info("Imports OK (%d modules, %d avertissements dépendances)", len(modules), len(warnings))
    return True


def test_text_normalizer() -> bool:
    """Teste la normalisation du texte."""
    logger.info("Test 2 : TextNormalizer...")
    try:
        from src.ingestion.normalizer import TextNormalizer
        norm = TextNormalizer()

        text = "Bonjour\x00 le  monde\r\ntest  "
        result = norm.normalize(text)
        assert "Bonjour" in result
        assert "\x00" not in result
        assert "  " not in result

        logger.info("  %s TextNormalizer OK", PASS)
        return True
    except Exception as e:
        logger.error("  %s TextNormalizer : %s", FAIL, e)
        return False


def test_qa_metrics() -> bool:
    """Teste les métriques QA (EM et F1)."""
    logger.info("Test 3 : QAMetrics...")
    try:
        from src.eval.qa_metrics import QAMetrics
        qa = QAMetrics()

        em = qa.exact_match("Paris", ["Paris", "paris"])
        assert em == 1, f"EM attendu 1, obtenu {em}"

        em_fail = qa.exact_match("Lyon", ["Paris"])
        assert em_fail == 0, f"EM attendu 0, obtenu {em_fail}"

        f1 = qa.token_f1("Le chat noir", ["Le chat"])
        assert 0.0 < f1 <= 1.0, f"F1 hors range : {f1}"

        batch = qa.compute_batch(["Paris", "Lyon"], [["Paris"], ["Paris"]])
        assert batch["em"] == 0.5
        assert 0.0 <= batch["f1"] <= 1.0

        logger.info("  %s QAMetrics OK", PASS)
        return True
    except Exception as e:
        logger.error("  %s QAMetrics : %s", FAIL, e)
        return False


def test_retrieval_metrics() -> bool:
    """Teste les métriques de retrieval."""
    logger.info("Test 4 : RetrievalMetrics...")
    try:
        from src.eval.retrieval_metrics import RetrievalMetrics
        rm = RetrievalMetrics()

        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc1", "doc3"}

        recall = rm.recall_at_k(retrieved, relevant, k=5)
        assert recall == 1.0, f"Recall@5 attendu 1.0, obtenu {recall}"

        recall_3 = rm.recall_at_k(retrieved, relevant, k=3)
        assert recall_3 == 1.0, f"Recall@3 attendu 1.0, obtenu {recall_3}"

        mrr = rm.mrr(retrieved, relevant)
        assert mrr == 1.0, f"MRR attendu 1.0, obtenu {mrr}"

        rel_scores = {"doc1": 1.0, "doc3": 0.5}
        ndcg = rm.ndcg_at_k(retrieved, rel_scores, k=5)
        assert 0.0 <= ndcg <= 1.0, f"nDCG hors range : {ndcg}"

        logger.info("  %s RetrievalMetrics OK", PASS)
        return True
    except Exception as e:
        logger.error("  %s RetrievalMetrics : %s", FAIL, e)
        return False


def test_statistical_tests() -> bool:
    """Teste les tests statistiques."""
    logger.info("Test 5 : StatisticalTests...")
    try:
        from src.eval.statistical_tests import StatisticalTests
        st = StatisticalTests()

        import numpy as np
        rng = np.random.default_rng(42)
        scores_a = rng.random(50).tolist()
        scores_b = (rng.random(50) + 0.2).tolist()

        perm = st.permutation_test(scores_a, scores_b, n_permutations=100)
        assert "p_value" in perm

        ci = st.bootstrap_ci(scores_a, n_bootstrap=100)
        assert ci["ci_lower"] <= ci["mean"] <= ci["ci_upper"]

        holm = st.holm_bonferroni([0.01, 0.05, 0.1, 0.2])
        assert len(holm["adjusted_p_values"]) == 4

        try:
            wilcoxon = st.wilcoxon_test(scores_a, scores_b)
            assert "p_value" in wilcoxon
            assert "stars" in wilcoxon
        except ImportError:
            logger.warning("  scipy non installé — wilcoxon_test ignoré (pip install scipy)")

        logger.info("  %s StatisticalTests OK", PASS)
        return True
    except Exception as e:
        logger.error("  %s StatisticalTests : %s", FAIL, e)
        return False


def test_prompt_formatter() -> bool:
    """Teste le formateur de prompts."""
    logger.info("Test 6 : PromptFormatter...")
    try:
        from src.rag.prompt_formatter import PromptFormatter
        formatter = PromptFormatter(config_path="configs/prompts.yaml")

        question = "Quelle est la capitale de la France ?"
        passages = ["Paris est la capitale de la France.", "Lyon est une grande ville."]

        standard_prompt = formatter.format_standard(question, passages)
        assert question in standard_prompt
        assert passages[0] in standard_prompt

        citations_prompt = formatter.format_citations(question, passages)
        assert "[1]" in citations_prompt
        assert "[2]" in citations_prompt

        baseline_prompt = formatter.format_baseline(question)
        assert question in baseline_prompt

        logger.info("  %s PromptFormatter OK", PASS)
        return True
    except Exception as e:
        logger.error("  %s PromptFormatter : %s", FAIL, e)
        return False


def test_context_builder() -> bool:
    """Teste le constructeur de contexte."""
    logger.info("Test 7 : ContextBuilder...")
    try:
        from src.analysis.error_taxonomy import ErrorTaxonomy, ErrorType
        from src.rag.context_builder import ContextBuilder
        from src.retrieval.bm25_retriever import RetrievedPassage
        builder = ContextBuilder()

        passages = [
            RetrievedPassage(rank=i + 1, passage_id=f"doc{i}", text=f"Texte {i}", bm25_score=float(10 - i))
            for i in range(5)
        ]

        ordered = builder.build(passages, gold_passage_id="doc0", position="first")
        assert ordered[0].passage_id == "doc0"

        ordered_last = builder.build(passages, gold_passage_id="doc0", position="last")
        assert ordered_last[-1].passage_id == "doc0"

        gold = passages[0]
        distractors = passages[1:]
        context = builder.build_with_distractors(
            gold_passage=gold,
            distractor_passages=distractors,
            k_total=5,
            distractor_ratio=0.5,
            position="first",
        )
        assert len(context) <= 5
        assert context[0].passage_id == "doc0"

        logger.info("  %s ContextBuilder OK", PASS)
        return True
    except Exception as e:
        logger.error("  %s ContextBuilder : %s", FAIL, e)
        return False


def test_error_taxonomy() -> bool:
    """Teste la classification des erreurs."""
    logger.info("Test 8 : ErrorTaxonomy...")
    try:
        from src.analysis.error_taxonomy import ErrorTaxonomy, ErrorType
        taxonomy = ErrorTaxonomy()

        retrieval_fail = {
            "metrics": {"em": 0, "f1": 0.0, "recall_at_10": 0.3, "faithfulness": 0.5}
        }
        assert taxonomy.classify(retrieval_fail) == ErrorType.RETRIEVAL_FAILURE

        unanswerable_item = {
            "metrics": {"em": 0, "f1": 0.0},
            "is_unanswerable": True,
            "generated_answer": "La réponse est Lyon.",
        }
        assert taxonomy.classify(unanswerable_item) == ErrorType.UNANSWERABLE_MISHANDLING

        items = [retrieval_fail, unanswerable_item]
        dist = taxonomy.compute_distribution(items)
        assert "total" in dist
        assert dist["total"] == 2

        logger.info("  %s ErrorTaxonomy OK", PASS)
        return True
    except Exception as e:
        logger.error("  %s ErrorTaxonomy : %s", FAIL, e)
        return False


def test_configs_exist() -> bool:
    """Vérifie l'existence de tous les fichiers de configuration."""
    logger.info("Test 9 : Fichiers de configuration...")
    required_files = [
        "configs/datasets.yaml",
        "configs/retrievers.yaml",
        "configs/generators.yaml",
        "configs/prompts.yaml",
        "requirements.txt",
        "README.md",
        "checksums.txt",
    ]
    missing = [f for f in required_files if not Path(f).exists()]
    if missing:
        logger.error("  %s Fichiers manquants : %s", FAIL, missing)
        return False
    logger.info("  %s Tous les fichiers de config présents", PASS)
    return True


def test_directory_structure() -> bool:
    """Vérifie que la structure de répertoires est correcte."""
    logger.info("Test 10 : Structure de répertoires...")
    required_dirs = [
        "src/ingestion", "src/retrieval", "src/rag", "src/eval", "src/analysis",
        "runs/A_ablation", "runs/B_context_noise", "runs/C_factuality",
        "reports/figures", "reports/tables", "reports/logs",
        "indexes/bm25", "indexes/dense", "indexes/hybrid",
        "data/raw", "data/processed", "data/splits",
        "annexes/prompts", "annexes/hyperparams", "annexes/qualitative_cases",
    ]
    missing = [d for d in required_dirs if not Path(d).exists()]
    if missing:
        logger.error("  %s Répertoires manquants : %s", FAIL, missing)
        return False
    logger.info("  %s Structure de répertoires correcte", PASS)
    return True


def main(fast: bool = False) -> None:
    start = time.time()
    logger.info("=" * 60)
    logger.info("SMOKE TEST — RAG Mémoire ADS CESI 2025-2026")
    logger.info("=" * 60)

    tests = [
        test_imports,
        test_text_normalizer,
        test_qa_metrics,
        test_retrieval_metrics,
        test_statistical_tests,
        test_prompt_formatter,
        test_context_builder,
        test_error_taxonomy,
        test_configs_exist,
        test_directory_structure,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            if test_fn():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error("Exception inattendue dans %s : %s", test_fn.__name__, e)
            failed += 1

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info("Résultats : %d/%d tests réussis — %.1fs", passed, passed + failed, elapsed)
    logger.info("=" * 60)

    if failed > 0:
        sys.exit(1)
    else:
        logger.info("Smoke test réussi — Projet prêt pour les expériences.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke test du projet RAG")
    parser.add_argument("--fast", action="store_true", help="Mode rapide (sans modèles lourds)")
    args = parser.parse_args()
    main(fast=args.fast)

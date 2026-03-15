"""Demo des modules RAG fonctionnels sans GPU."""

from __future__ import annotations

import sys
sys.path.insert(0, ".")

print("=" * 50)
print("Demo RAG Mémoire — ADS CESI 2025-2026")
print("=" * 50)

# ── 1. Normalisation ──────────────────────────────────
from src.ingestion.normalizer import TextNormalizer

norm = TextNormalizer()
raw = "Rennes est la  capitale\x00 de la Bretagne...\r\n  test  "
print("\n[1] Normalizer")
print("  Avant :", repr(raw[:40]))
print("  Après :", repr(norm.normalize(raw)))

# ── 2. Métriques QA ──────────────────────────────────
from src.eval.qa_metrics import QAMetrics

qa = QAMetrics()
print("\n[2] QA Metrics")
print("  EM('Rennes', ['Rennes'])          =", qa.exact_match("Rennes", ["Rennes"]))
print("  EM('rennes', ['Rennes'])           =", qa.exact_match("rennes", ["Rennes"]))
print("  F1('capitale Bretagne', ['Rennes capitale de la Bretagne']) =",
      round(qa.token_f1("capitale Bretagne", ["Rennes capitale de la Bretagne"]), 3))

batch = qa.compute_batch(
    ["Paris", "Lyon", "paris"],
    [["Paris"], ["Paris"], ["Paris"]],
)
print("  Batch EM =", batch["em"], "| F1 =", round(batch["f1"], 3))

# ── 3. Métriques Retrieval ───────────────────────────
from src.eval.retrieval_metrics import RetrievalMetrics

rm = RetrievalMetrics()
retrieved = ["doc_gold", "doc2", "doc3", "doc4", "doc5", "doc6", "doc7", "doc8", "doc9", "doc10"]
relevant = {"doc_gold"}

print("\n[3] Retrieval Metrics")
print("  Recall@1  =", rm.recall_at_k(retrieved, relevant, k=1))
print("  Recall@5  =", rm.recall_at_k(retrieved, relevant, k=5))
print("  Recall@10 =", rm.recall_at_k(retrieved, relevant, k=10))
print("  MRR       =", rm.mrr(retrieved, relevant))
print("  nDCG@10   =", round(rm.ndcg_at_k(retrieved, {"doc_gold": 1.0}, k=10), 3))

# ── 4. Prompts ───────────────────────────────────────
from src.rag.prompt_formatter import PromptFormatter

formatter = PromptFormatter()
question = "Quelle est la capitale de la Bretagne ?"
passages = [
    "Rennes est la capitale de la région Bretagne depuis 1972.",
    "La Bretagne est une région française au nord-ouest.",
]

print("\n[4] Prompt Standard")
print(formatter.format_standard(question, passages))

print("\n[4] Prompt Citations")
print(formatter.format_citations(question, passages))

# ── 5. Context Builder (position) ───────────────────
from src.rag.context_builder import ContextBuilder
from src.retrieval.bm25_retriever import RetrievedPassage

builder = ContextBuilder()
psg = [
    RetrievedPassage(rank=i + 1, passage_id=f"p{i}", text=f"Passage {i}", bm25_score=float(10 - i))
    for i in range(5)
]

print("\n[5] Context Builder")
print("  Ordre original :", [p.passage_id for p in psg])

reordered_last = builder.build(psg, gold_passage_id="p0", position="last")
print("  Gold en LAST   :", [p.passage_id for p in reordered_last])

reordered_mid = builder.build(psg, gold_passage_id="p0", position="middle")
print("  Gold au MILIEU :", [p.passage_id for p in reordered_mid])

# Avec distracteurs (Protocole B)
gold = psg[0]
distractors = psg[1:]
ctx = builder.build_with_distractors(gold, distractors, k_total=5, distractor_ratio=0.4, position="first")
print("  Contexte p=0.4 :", [p.passage_id for p in ctx])

# ── 6. Tests Statistiques ────────────────────────────
import numpy as np
from src.eval.statistical_tests import StatisticalTests

st = StatisticalTests()
rng = np.random.default_rng(42)
scores_a = rng.random(50).tolist()
scores_b = (rng.random(50) + 0.15).tolist()

perm = st.permutation_test(scores_a, scores_b, n_permutations=2000)
ci_a = st.bootstrap_ci(scores_a, n_bootstrap=1000)
ci_b = st.bootstrap_ci(scores_b, n_bootstrap=1000)
holm = st.holm_bonferroni([0.001, 0.004, 0.03, 0.07, 0.15, 0.25, 0.4, 0.8])

try:
    wilcox = st.wilcoxon_test(scores_a, scores_b)
    wilcox_str = f"p={wilcox['p_value']:.3f} {wilcox['stars']}"
except ImportError:
    wilcox_str = "scipy non installé"

print("\n[6] Tests Statistiques (n=50 par système)")
print(f"  Permutation   : diff_obs={perm['observed_diff']:.3f}, p={perm['p_value']:.3f} {perm['stars']}")
print(f"  Bootstrap A   : {ci_a['mean']:.3f} IC95[{ci_a['ci_lower']:.3f}, {ci_a['ci_upper']:.3f}]")
print(f"  Bootstrap B   : {ci_b['mean']:.3f} IC95[{ci_b['ci_lower']:.3f}, {ci_b['ci_upper']:.3f}]")
print(f"  Wilcoxon      : {wilcox_str}")
print(f"  Holm-Bonf.    : {sum(holm['rejected'])}/{holm['n_comparisons']} comparaisons sig. (alpha=0.05)")

# ── 7. Taxonomie des erreurs ─────────────────────────
from src.analysis.error_taxonomy import ErrorTaxonomy, ErrorType

taxonomy = ErrorTaxonomy()
cases = [
    {"metrics": {"em": 0, "f1": 0.0, "recall_at_10": 0.2, "faithfulness": 0.4}},
    {"metrics": {"em": 0, "f1": 0.1, "recall_at_10": 0.9, "faithfulness": 0.3}},
    {"metrics": {"em": 1, "f1": 0.5, "recall_at_10": 0.8, "faithfulness": 0.4}},
    {"metrics": {"em": 0, "f1": 0.0}, "is_unanswerable": True, "generated_answer": "La réponse est Paris."},
    {"metrics": {"em": 1, "f1": 1.0, "recall_at_10": 1.0, "faithfulness": 0.95}},
    {"metrics": {"em": 0, "f1": 0.0, "recall_at_10": 0.3, "faithfulness": 0.5}},
    {"metrics": {"em": 0, "f1": 0.0}, "is_unanswerable": True, "generated_answer": "Je ne sais pas."},
]

dist = taxonomy.compute_distribution(cases)
print("\n[7] Taxonomie des Erreurs (7 cas)")
for et, info in dist["distribution"].items():
    print(f"  {et:<35} : {info['count']} cas ({info['percentage']}%)")

print("\n" + "=" * 50)
print("Demo réussie — Tous les modules opérationnels.")
print("=" * 50)

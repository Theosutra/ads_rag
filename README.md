# RAG : Qualité de récupération et factualité

> Mémoire ADS — CESI École d'Ingénieurs 2025-2026  
> Auteur : Theo (Datasulting)

## Problématique scientifique

> Dans un système RAG reposant sur une architecture générative fixe, comment la performance globale et la fiabilité factuelle évoluent-elles lorsque l'on modifie la qualité du mécanisme de récupération de l'information ou la manière dont le contexte est construit avant la génération de la réponse ?

## Architecture

```
Question → Retriever (BM25 / Dense / Hybride)
         → Reranker (ColBERT / Cross-encodeur, optionnel)
         → Construction contexte (standard / citations)
         → Générateur (mT5-large, FIXE)
         → Évaluation (EM, F1, Faithfulness)
```

**Principe clé** : le modèle génératif est maintenu constant. Seuls le retriever, le reranker et la construction du contexte varient.

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Prérequis matériel

- GPU A100 40GB (retrieval dense + génération)
- CPU 32 cœurs (BM25 + expériences légères)
- Python 3.11

## Reproductibilité — Checklist

```
[ ] Python 3.11 installé
[ ] pip install -r requirements.lock (versions verrouillées)
[ ] Seeds fixées (42) dans tous les modules
[ ] SHA256 des datasets vérifiés (checksums.txt)
[ ] Matériel déclaré : GPU A100 40GB (dense), CPU 32 cœurs (BM25)
[ ] Index versionnés : date + chunk_size + overlap + modèle embedding
[ ] Logs complets JSONL conservés
[ ] Commandes exactes dans run_experiments.sh
[ ] Smoke test : python -m tests.smoke_test --fast (≤ 5 min)
```

## Structure du projet

```
rag-memoire/
├── configs/              # Hyperparamètres YAML
├── data/                 # Données brutes, traitées, splits
├── indexes/              # Index BM25, FAISS, hybride
├── src/
│   ├── ingestion/        # Chargement, normalisation, chunking
│   ├── retrieval/        # BM25, dense, hybride, reranker
│   ├── rag/              # Construction contexte, prompt, générateur
│   ├── eval/             # EM/F1, métriques retrieval, RAGAS, tests stat
│   └── analysis/         # Taxonomie erreurs, visualisations
├── runs/                 # Résultats JSONL par protocole
├── reports/              # Figures, tableaux, logs
├── annexes/              # Prompts, hyperparams, cas annotés
└── tests/                # Smoke test
```

## Protocoles expérimentaux

### Protocole A — Ablation end-to-end bilingue
Teste H1 (saturation k=10) et H2 (reranking → faithfulness).  
Plan factoriel : retriever × k × reranking × format × langue.

### Protocole B — Dégradation contrôlée du contexte
Teste H1 (seuil bruit) et H3 (effet position).  
Fait varier la proportion de distracteurs p ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}.

### Protocole C — Factualité fine et attribution
Teste H2 et H3 sur les hallucinations résiduelles.  
4 configs : RAG_standard, RAG_rerank, RAG_citations, RAG_verify.

## Exécution

```bash
# 1. Vérifier checksums
sha256sum --check checksums.txt

# 2. Construire les index
python src/ingestion/build_indexes.py --config configs/datasets.yaml

# 3. Protocole A
bash run_experiments.sh --protocol A --lang fr
bash run_experiments.sh --protocol A --lang en

# 4. Protocoles B et C
bash run_experiments.sh --protocol B
bash run_experiments.sh --protocol C

# 5. Calculer toutes les métriques
python src/eval/compute_all_metrics.py --run runs/A_ablation/
python src/eval/compute_all_metrics.py --run runs/B_context_noise/
python src/eval/compute_all_metrics.py --run runs/C_factuality/

# 6. Smoke test (< 5 min)
python -m tests.smoke_test --fast
```

## Hypothèses à tester

| ID | Hypothèse |
|----|-----------|
| H1 | L'amélioration de la qualité de récupération augmente les performances jusqu'à un point de saturation (~10 passages). Au-delà, le bruit dégrade la factualité. |
| H2 | Le reranking améliore prioritairement la fidélité (Faithfulness) plutôt que les métriques EM/F1. |
| H3 | La construction du contexte (ordre, format, taille chunk) a un impact comparable à celui du choix du retriever sur la factualité. |
| H4 | Les retrievers denses/hybrides ont une variabilité plus élevée sur données françaises qu'anglaises. Les hybrides restent les plus robustes bilingues. |

## Datasets

| Dataset | Langue | Usage |
|---------|--------|-------|
| FQuAD 2.0 | FR | Protocoles A, B, C |
| PIAF | FR | Protocole A (généralisation) |
| KILT Natural Questions | EN | Protocoles A, B |
| KILT Wikipedia | FR+EN | Corpus documentaire |
| RAGTruth | EN | Protocole C (hallucinations) |
| RAGBench | EN | Protocole C (éval industrielle) |

## Références

- Lewis et al. (2020) — RAG original — NeurIPS 2020
- Karpukhin et al. (2020) — DPR — EMNLP 2020
- Khattab & Zaharia (2020) — ColBERT — SIGIR 2020
- Petroni et al. (2021) — KILT — NAACL 2021
- Es et al. (2023) — RAGAS — arXiv:2309.15217
- Niu et al. (2024) — RAGTruth — ACL 2024
- Xue et al. (2021) — mT5 — NAACL 2021

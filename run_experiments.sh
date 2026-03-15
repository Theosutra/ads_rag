#!/bin/bash
# run_experiments.sh — Commandes exactes pour reproduire les expériences RAG
# Usage : bash run_experiments.sh --protocol A --lang fr

set -e

# ============================================================
# Seeds de reproductibilité
# ============================================================
export RANDOM_SEED=42
export TORCH_SEED=42
export NUMPY_SEED=42

# ============================================================
# Parsing des arguments
# ============================================================
PROTOCOL="A"
LANG="fr"
MAX_SAMPLES=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --protocol) PROTOCOL="$2"; shift ;;
        --lang) LANG="$2"; shift ;;
        --max-samples) MAX_SAMPLES="--max-samples $2"; shift ;;
        *) echo "Argument inconnu : $1"; exit 1 ;;
    esac
    shift
done

echo "============================================================"
echo "RAG Mémoire — Protocole $PROTOCOL — Langue $LANG"
echo "============================================================"

# ============================================================
# 1. Vérification des checksums
# ============================================================
if [ -f "checksums.txt" ] && grep -q "[a-f0-9]\{64\}" checksums.txt 2>/dev/null; then
    echo "[1] Vérification des checksums..."
    sha256sum --check checksums.txt || echo "AVERTISSEMENT : Certains checksums ne correspondent pas."
else
    echo "[1] checksums.txt non peuplé — ignorer pour l'instant."
fi

# ============================================================
# 2. Construction des index (si nécessaire)
# ============================================================
if [ "$PROTOCOL" = "A" ] || [ "$PROTOCOL" = "B" ] || [ "$PROTOCOL" = "C" ]; then
    if [ ! -d "indexes/bm25/index.properties" ] 2>/dev/null; then
        echo "[2] Construction des index..."
        python src/ingestion/build_indexes.py --config configs/datasets.yaml
    else
        echo "[2] Index déjà construits — ignoré."
    fi
fi

# ============================================================
# 3. Exécution du protocole
# ============================================================
echo "[3] Exécution Protocole $PROTOCOL — Langue $LANG..."
python run_experiments.py --protocol "$PROTOCOL" --lang "$LANG" $MAX_SAMPLES

# ============================================================
# 4. Calcul des métriques
# ============================================================
echo "[4] Calcul des métriques..."
case $PROTOCOL in
    A) RUN_DIR="runs/A_ablation/" ;;
    B) RUN_DIR="runs/B_context_noise/" ;;
    C) RUN_DIR="runs/C_factuality/" ;;
    *) RUN_DIR="runs/" ;;
esac

python src/eval/compute_all_metrics.py --run "$RUN_DIR" --output "reports/tables/"

echo "============================================================"
echo "Protocole $PROTOCOL terminé. Résultats dans $RUN_DIR"
echo "Métriques dans reports/tables/"
echo "============================================================"

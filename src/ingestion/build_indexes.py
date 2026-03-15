"""Script de construction de tous les index (BM25, Dense, Hybride)."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ingestion.chunker import DocumentChunker
from src.ingestion.loader import DatasetLoader
from src.ingestion.normalizer import TextNormalizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)


def build_bm25_index(processed_dir: str, index_dir: str, threads: int = 8) -> None:
    """Construit l'index BM25 via Pyserini/Lucene."""
    logger.info("Construction index BM25 → %s", index_dir)
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", processed_dir,
        "--index", index_dir,
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", str(threads),
        "--storePositions",
        "--storeDocvectors",
        "--storeRaw",
    ]
    logger.info("Commande : %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("Erreur BM25 : %s", result.stderr)
        raise RuntimeError(f"Échec construction index BM25 : {result.stderr}")
    logger.info("Index BM25 construit avec succès")


def prepare_corpus_for_pyserini(passages: list, output_dir: str) -> None:
    """Prépare le corpus au format JsonCollection pour Pyserini."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    batch_size = 10000
    for batch_idx in range(0, len(passages), batch_size):
        batch = passages[batch_idx: batch_idx + batch_size]
        filename = output_path / f"passages_{batch_idx:08d}.jsonl"
        with open(filename, "w", encoding="utf-8") as f:
            for p in batch:
                doc = {"id": p.id, "contents": p.text}
                doc.update(p.metadata)
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    logger.info("Corpus Pyserini prêt : %d passages → %s", len(passages), output_dir)


def main(config_path: str) -> None:
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    loader = DatasetLoader(config_path)
    normalizer = TextNormalizer()
    chunker = DocumentChunker(chunk_size=256, overlap=50)

    logger.info("=== Chargement et normalisation du corpus KILT ===")
    kilt_nq = loader.load_kilt_nq("validation")

    documents = []
    for sample in kilt_nq:
        if sample.get("context"):
            doc = normalizer.normalize_document({
                "id": f"kilt_{sample['id']}",
                "text": sample["context"],
                "lang": sample["lang"],
                "source": sample["source"],
            })
            documents.append(doc)

    logger.info("%d documents chargés", len(documents))

    passages = chunker.chunk_corpus(documents)
    logger.info("%d passages générés", len(passages))

    processed_dir = config.get("processed_dir", "data/processed")
    prepare_corpus_for_pyserini(passages, processed_dir)

    bm25_index_dir = "indexes/bm25"
    build_bm25_index(processed_dir, bm25_index_dir)

    logger.info("=== Construction des index terminée ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construction des index RAG")
    parser.add_argument("--config", default="configs/datasets.yaml")
    args = parser.parse_args()
    main(args.config)

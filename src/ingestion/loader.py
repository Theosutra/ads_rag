"""Chargement et préparation des datasets pour les expériences RAG."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import jsonlines
import yaml
from datasets import load_dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Charge les datasets depuis HuggingFace ou le disque local."""

    def __init__(self, config_path: str = "configs/datasets.yaml") -> None:
        with open(config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.raw_dir = Path(self.config["raw_dir"])
        self.processed_dir = Path(self.config["processed_dir"])
        self.splits_dir = Path(self.config["splits_dir"])

    def load_fquad2(self, split: str = "validation") -> list[dict[str, Any]]:
        """Charge FQuAD 2.0 depuis HuggingFace."""
        logger.info("Chargement FQuAD 2.0 — split: %s", split)
        ds = load_dataset("illuin-technology/fquad2", split=split, trust_remote_code=True)
        samples = []
        for item in tqdm(ds, desc="FQuAD 2.0"):
            answers = item.get("answers", {})
            answer_texts = answers.get("text", []) if isinstance(answers, dict) else []
            samples.append({
                "id": item.get("id", ""),
                "question": item["question"],
                "context": item.get("context", ""),
                "answers": answer_texts,
                "is_impossible": item.get("is_impossible", False),
                "lang": "fr",
                "source": "fquad2",
            })
        return samples

    def load_piaf(self, split: str = "train") -> list[dict[str, Any]]:
        """Charge PIAF depuis HuggingFace."""
        logger.info("Chargement PIAF — split: %s", split)
        ds = load_dataset("illuin-technology/piaf", split=split, trust_remote_code=True)
        samples = []
        for item in tqdm(ds, desc="PIAF"):
            answers = item.get("answers", {})
            answer_texts = answers.get("text", []) if isinstance(answers, dict) else []
            samples.append({
                "id": item.get("id", ""),
                "question": item["question"],
                "context": item.get("context", ""),
                "answers": answer_texts,
                "is_impossible": False,
                "lang": "fr",
                "source": "piaf",
            })
        return samples

    def load_kilt_nq(self, split: str = "validation") -> list[dict[str, Any]]:
        """Charge KILT Natural Questions."""
        logger.info("Chargement KILT NQ — split: %s", split)
        ds = load_dataset("kilt_tasks", "nq", split=split, trust_remote_code=True)
        samples = []
        for item in tqdm(ds, desc="KILT NQ"):
            output = item.get("output", [])
            answers = [o["answer"] for o in output if o.get("answer")]
            provenance = [
                {"wikipedia_id": p.get("wikipedia_id", ""), "start_paragraph_id": p.get("start_paragraph_id", 0)}
                for o in output for p in o.get("provenance", [])
            ]
            samples.append({
                "id": item["id"],
                "question": item["input"],
                "answers": answers,
                "provenance": provenance,
                "is_impossible": len(answers) == 0,
                "lang": "en",
                "source": "kilt_nq",
            })
        return samples

    def load_from_jsonl(self, path: str) -> list[dict[str, Any]]:
        """Charge des données depuis un fichier JSONL."""
        samples = []
        with jsonlines.open(path) as reader:
            for item in reader:
                samples.append(item)
        return samples

    def save_to_jsonl(self, data: list[dict[str, Any]], path: str) -> None:
        """Sauvegarde des données au format JSONL."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with jsonlines.open(path, mode="w") as writer:
            writer.write_all(data)
        logger.info("Sauvegardé %d échantillons → %s", len(data), path)

    def save_to_json(self, data: list[dict[str, Any]], path: str) -> None:
        """Sauvegarde des données au format JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("Sauvegardé %d échantillons → %s", len(data), path)

    def prepare_all_datasets(self) -> None:
        """Télécharge et sauvegarde tous les datasets."""
        logger.info("=== Préparation de tous les datasets ===")

        fquad2_dev = self.load_fquad2("validation")
        self.save_to_jsonl(fquad2_dev, self.raw_dir / "fquad2_dev.jsonl")

        fquad2_train = self.load_fquad2("train")
        self.save_to_jsonl(fquad2_train, self.raw_dir / "fquad2_train.jsonl")

        piaf = self.load_piaf("train")
        self.save_to_jsonl(piaf, self.raw_dir / "piaf_v1.1.jsonl")

        kilt_nq_dev = self.load_kilt_nq("validation")
        self.save_to_jsonl(kilt_nq_dev, self.raw_dir / "kilt_nq_dev.jsonl")

        kilt_nq_train = self.load_kilt_nq("train")
        self.save_to_jsonl(kilt_nq_train, self.raw_dir / "kilt_nq_train.jsonl")

        logger.info("=== Tous les datasets sont prêts ===")

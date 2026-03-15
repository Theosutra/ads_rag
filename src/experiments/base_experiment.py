"""Classe de base pour tous les protocoles expérimentaux."""

from __future__ import annotations

import datetime
import json
import logging
import random
from pathlib import Path
from typing import Any

import jsonlines
import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)

SEED = 42


def set_all_seeds(seed: int = SEED) -> None:
    """Fixe toutes les graines pour la reproductibilité totale."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BaseExperiment:
    """
    Classe de base pour les expériences RAG.

    Gère le logging JSONL, les seeds, et la persistance des résultats.
    """

    def __init__(
        self,
        output_dir: str,
        config_dir: str = "configs/",
        seed: int = SEED,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir = Path(config_dir)
        self.seed = seed
        set_all_seeds(seed)

        self._retriever_config = self._load_yaml("retrievers.yaml")
        self._generator_config = self._load_yaml("generators.yaml")

    def _load_yaml(self, filename: str) -> dict:
        path = self.config_dir / filename
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _make_run_id(self, protocol: str, config_name: str, lang: str) -> str:
        ts = datetime.datetime.utcnow().strftime("%Y%m%d%H%M")
        return f"{protocol}_{config_name}_{lang}_{ts}"

    def log_result(
        self,
        run_file: Path,
        question_id: str,
        question: str,
        gold_answer: str | list[str],
        config: dict[str, Any],
        retrieved_passages: list[dict[str, Any]],
        prompt: str,
        generated_answer: str,
        metrics: dict[str, float],
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Enregistre un résultat au format JSONL standard du mémoire.

        Format conforme à la spécification section 11.
        """
        record = {
            "run_id": self._make_run_id(
                config.get("protocol", "X"),
                config.get("retriever", "baseline"),
                config.get("lang", "fr"),
            ),
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "question_id": question_id,
            "question": question,
            "gold_answer": gold_answer,
            "config": config,
            "retrieved_passages": retrieved_passages,
            "prompt": prompt,
            "generated_answer": generated_answer,
            "metrics": metrics,
        }
        if extra:
            record.update(extra)

        with jsonlines.open(run_file, mode="a") as writer:
            writer.write(record)

        return record

    def get_run_file(self, config_name: str, lang: str) -> Path:
        """Retourne le chemin du fichier de log pour une configuration."""
        filename = f"run_{config_name}_{lang}.jsonl"
        return self.output_dir / filename

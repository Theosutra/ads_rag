"""Wrapper mT5-large pour la génération de réponses RAG."""

from __future__ import annotations

import logging
import random

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)

MODEL_NAME = "google/mt5-large"

GENERATION_PARAMS = {
    "max_new_tokens": 128,
    "num_beams": 4,
    "early_stopping": True,
    "no_repeat_ngram_size": 3,
}

SEED = 42


def set_seeds(seed: int = SEED) -> None:
    """Fixe toutes les graines pour la reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RAGGenerator:
    """
    Wrapper autour de mT5-large pour la génération de réponses.

    Le modèle est MAINTENU CONSTANT dans toutes les expériences.
    Seuls le retriever, le reranker et la construction du contexte varient.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str | None = None,
        dtype: str = "float16",
        generation_params: dict | None = None,
        seed: int = SEED,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if dtype == "float16" and self.device == "cuda" else torch.float32
        self.generation_params = generation_params or GENERATION_PARAMS.copy()
        self.seed = seed

        self._model = None
        self._tokenizer = None

        set_seeds(seed)

    def _load_model(self) -> None:
        """Charge le modèle mT5 (lazy loading)."""
        logger.info("Chargement %s sur %s (dtype=%s)", self.model_name, self.device, self.dtype)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
        ).to(self.device)
        self._model.eval()
        logger.info("Modèle chargé : %s", self.model_name)

    def generate(self, prompt: str) -> str:
        """
        Génère une réponse pour un prompt donné.

        Args:
            prompt: prompt formaté avec contexte et question

        Returns:
            Réponse générée
        """
        if self._model is None:
            self._load_model()

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                **self.generation_params,
            )

        answer = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return answer.strip()

    def generate_batch(self, prompts: list[str], batch_size: int = 8) -> list[str]:
        """
        Génère des réponses pour un lot de prompts.

        Args:
            prompts: liste de prompts formatés
            batch_size: taille du batch

        Returns:
            Liste de réponses générées
        """
        if self._model is None:
            self._load_model()

        answers = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i: i + batch_size]
            inputs = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            ).to(self.device)

            with torch.no_grad():
                output_ids = self._model.generate(**inputs, **self.generation_params)

            for ids in output_ids:
                answer = self._tokenizer.decode(ids, skip_special_tokens=True)
                answers.append(answer.strip())

        return answers

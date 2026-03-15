"""Formats de prompt : standard (concaténation brute) et citations numérotées."""

from __future__ import annotations

import logging

import yaml

logger = logging.getLogger(__name__)

PROMPT_STANDARD = """\
Utilise les passages suivants pour répondre à la question.
Si la réponse ne figure pas dans les passages, réponds 'Je ne sais pas'.

Passages :
{passages}

Question : {question}
Réponse :"""

PROMPT_CITATIONS = """\
Réponds à la question en te basant UNIQUEMENT sur les passages numérotés ci-dessous.
Pour chaque affirmation, indique le numéro du passage source entre crochets [N].
Si aucun passage ne permet de répondre, réponds exactement 'Je ne sais pas.'

{numbered_passages}

Question : {question}
Réponse (avec citations) :"""


class PromptFormatter:
    """
    Formate le prompt final pour le générateur mT5.

    Deux formats disponibles :
    - standard : concaténation brute des passages
    - citations : passages numérotés avec instruction d'attribution
    """

    def __init__(self, config_path: str = "configs/prompts.yaml") -> None:
        try:
            with open(config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            prompts = cfg.get("prompts", {})
            self._standard_template = prompts.get("standard", {}).get("template", PROMPT_STANDARD).strip()
            self._citations_template = prompts.get("citations", {}).get("template", PROMPT_CITATIONS).strip()
            self.unanswerable_marker = prompts.get("unanswerable_marker", "Je ne sais pas")
        except FileNotFoundError:
            self._standard_template = PROMPT_STANDARD
            self._citations_template = PROMPT_CITATIONS
            self.unanswerable_marker = "Je ne sais pas"

    def format_standard(self, question: str, passages: list[str]) -> str:
        """
        Formate le prompt avec concaténation brute des passages.

        Args:
            question: question en langage naturel
            passages: liste des textes de passages

        Returns:
            Prompt formaté
        """
        passages_text = "\n".join(passages)
        return self._standard_template.format(passages=passages_text, question=question)

    def format_citations(self, question: str, passages: list[str]) -> str:
        """
        Formate le prompt avec citations numérotées.

        Args:
            question: question en langage naturel
            passages: liste des textes de passages

        Returns:
            Prompt formaté avec numéros [1], [2], ...
        """
        numbered = "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
        return self._citations_template.format(numbered_passages=numbered, question=question)

    def format(self, question: str, passages: list[str], fmt: str = "standard") -> str:
        """
        Formate le prompt selon le format choisi.

        Args:
            question: question en langage naturel
            passages: liste des textes de passages
            fmt: 'standard' ou 'citations'

        Returns:
            Prompt formaté
        """
        if fmt == "citations":
            return self.format_citations(question, passages)
        if fmt == "standard":
            return self.format_standard(question, passages)
        raise ValueError(f"Format inconnu : {fmt}. Choisir 'standard' ou 'citations'.")

    def format_baseline(self, question: str) -> str:
        """Formate un prompt baseline sans contexte récupéré."""
        return f"Question : {question}\nRéponse :"

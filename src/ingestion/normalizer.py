"""Normalisation du texte pour les documents du corpus RAG."""

from __future__ import annotations

import re
import unicodedata


class TextNormalizer:
    """
    Normalise le texte brut avant chunking et indexation.

    Opérations appliquées :
    - Suppression des caractères de contrôle
    - Normalisation Unicode (NFC)
    - Suppression des espaces multiples
    - Note : pas de suppression des stopwords (nécessaire pour BM25 francophone)
    """

    _CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
    _MULTI_SPACES_RE = re.compile(r" {2,}")
    _MULTI_NEWLINES_RE = re.compile(r"\n{3,}")

    def normalize(self, text: str) -> str:
        """
        Normalise un texte brut.

        Args:
            text: texte brut à normaliser

        Returns:
            Texte normalisé
        """
        if not text:
            return ""

        text = unicodedata.normalize("NFC", text)
        text = self._CONTROL_CHARS_RE.sub("", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = self._MULTI_SPACES_RE.sub(" ", text)
        text = self._MULTI_NEWLINES_RE.sub("\n\n", text)
        text = text.strip()
        return text

    def normalize_batch(self, texts: list[str]) -> list[str]:
        """Normalise une liste de textes."""
        return [self.normalize(t) for t in texts]

    def normalize_document(self, doc: dict) -> dict:
        """
        Normalise les champs textuels d'un document.

        Args:
            doc: dictionnaire avec au moins un champ 'text' ou 'context'

        Returns:
            Document avec les champs textuels normalisés
        """
        result = doc.copy()
        for field in ("text", "context", "title", "question"):
            if field in result and result[field]:
                result[field] = self.normalize(result[field])
        return result

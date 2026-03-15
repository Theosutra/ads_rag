"""Découpage des documents en passages homogènes pour l'indexation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

DEFAULT_TOKENIZER = "intfloat/multilingual-e5-large"


@dataclass
class Passage:
    """Un passage issu d'un document source."""

    id: str
    text: str
    doc_id: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
        }


class DocumentChunker:
    """
    Découpe les documents en segments de taille homogène.

    Valeurs retenues : chunk_size=256 tokens, overlap=50 tokens.
    """

    def __init__(
        self,
        chunk_size: int = 256,
        overlap: int = 50,
        tokenizer_name: str = DEFAULT_TOKENIZER,
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        logger.info("Chargement tokenizer : %s", tokenizer_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def chunk_document(self, text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
        """
        Découpe un document en segments de taille homogène.

        Args:
            text: texte normalisé du document
            chunk_size: taille cible en tokens (défaut : self.chunk_size)
            overlap: chevauchement entre segments successifs (défaut : self.overlap)

        Returns:
            Liste de passages textuels
        """
        chunk_size = chunk_size or self.chunk_size
        overlap = overlap if overlap is not None else self.overlap

        if not text:
            return []

        token_ids = self.tokenizer.encode(text, add_special_tokens=False)

        if len(token_ids) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(token_ids):
            end = min(start + chunk_size, len(token_ids))
            chunk_ids = token_ids[start:end]
            chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True)
            chunks.append(chunk_text.strip())
            if end == len(token_ids):
                break
            start += chunk_size - overlap

        return [c for c in chunks if c]

    def chunk_document_to_passages(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[Passage]:
        """
        Découpe un document en objets Passage avec identifiants.

        Args:
            doc_id: identifiant du document source
            text: texte normalisé
            metadata: métadonnées du document (titre, langue, source, etc.)

        Returns:
            Liste d'objets Passage
        """
        chunks = self.chunk_document(text)
        passages = []
        for i, chunk_text in enumerate(chunks):
            passage_id = f"{doc_id}_chunk_{i}"
            passages.append(Passage(
                id=passage_id,
                text=chunk_text,
                doc_id=doc_id,
                chunk_index=i,
                metadata=metadata or {},
            ))
        return passages

    def chunk_corpus(
        self,
        documents: list[dict[str, Any]],
        text_field: str = "text",
        id_field: str = "id",
    ) -> list[Passage]:
        """
        Découpe un corpus entier en passages.

        Args:
            documents: liste de documents avec au moins les champs id et text
            text_field: nom du champ contenant le texte
            id_field: nom du champ contenant l'identifiant

        Returns:
            Liste de tous les passages du corpus
        """
        all_passages = []
        for doc in documents:
            doc_id = doc.get(id_field, "")
            text = doc.get(text_field, "")
            metadata = {k: v for k, v in doc.items() if k not in (text_field, id_field)}
            passages = self.chunk_document_to_passages(doc_id, text, metadata)
            all_passages.extend(passages)
        logger.info(
            "Chunking terminé : %d documents → %d passages (chunk_size=%d, overlap=%d)",
            len(documents), len(all_passages), self.chunk_size, self.overlap,
        )
        return all_passages

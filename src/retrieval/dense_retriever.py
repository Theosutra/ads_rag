"""Retriever dense via encodeurs SBERT/DPR et index FAISS."""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, DPRQuestionEncoder

from .bm25_retriever import RetrievedPassage

logger = logging.getLogger(__name__)

MODELS = {
    "fr": "intfloat/multilingual-e5-large",
    "en": "facebook/dpr-ctx_encoder-single-nq-base",
}

QUERY_MODELS = {
    "fr": "intfloat/multilingual-e5-large",
    "en": "facebook/dpr-question_encoder-single-nq-base",
}


class DenseRetriever:
    """
    Retriever dense basé sur des encodeurs neuronaux et un index FAISS.

    - FR : intfloat/multilingual-e5-large
    - EN : facebook/dpr-ctx_encoder-single-nq-base
    """

    def __init__(
        self,
        index_dir: str = "indexes/dense/",
        lang: str = "fr",
        model_name: str | None = None,
        query_model_name: str | None = None,
        device: str | None = None,
        batch_size: int = 64,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.lang = lang
        self.model_name = model_name or MODELS[lang]
        self.query_model_name = query_model_name or QUERY_MODELS[lang]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        self._passage_encoder = None
        self._query_encoder = None
        self._tokenizer = None
        self._query_tokenizer = None
        self._index = None
        self._passage_ids: list[str] = []
        self._passage_texts: list[str] = []

    def _load_encoders(self) -> None:
        """Charge les encodeurs (lazy loading)."""
        logger.info("Chargement encodeurs — modèle : %s", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._passage_encoder = AutoModel.from_pretrained(self.model_name).to(self.device)
        self._passage_encoder.eval()

        if self.query_model_name != self.model_name:
            self._query_tokenizer = AutoTokenizer.from_pretrained(self.query_model_name)
            self._query_encoder = AutoModel.from_pretrained(self.query_model_name).to(self.device)
            self._query_encoder.eval()
        else:
            self._query_tokenizer = self._tokenizer
            self._query_encoder = self._passage_encoder

    def _encode_texts(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """Encode une liste de textes en vecteurs denses."""
        tokenizer = self._query_tokenizer if is_query else self._tokenizer
        model = self._query_encoder if is_query else self._passage_encoder

        if self.lang == "fr" or "e5" in self.model_name.lower():
            prefix = "query: " if is_query else "passage: "
            texts = [prefix + t for t in texts]

        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                embeddings = outputs.pooler_output
            else:
                embeddings = outputs.last_hidden_state[:, 0, :]

            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    def build_index(self, passages: list[dict[str, Any]], nlist: int = 4096) -> None:
        """
        Construit l'index FAISS à partir des passages.

        Utilise IndexIVFFlat pour les grands corpus (> 100K passages).
        """
        try:
            import faiss
        except ImportError as e:
            raise ImportError("faiss-cpu requis. pip install faiss-cpu") from e

        if self._passage_encoder is None:
            self._load_encoders()

        logger.info("Encodage de %d passages...", len(passages))
        texts = [p["text"] for p in passages]
        self._passage_ids = [p["id"] for p in passages]
        self._passage_texts = texts

        embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Encodage passages"):
            batch = texts[i: i + self.batch_size]
            emb = self._encode_texts(batch, is_query=False)
            embeddings.append(emb)

        all_embeddings = np.vstack(embeddings).astype(np.float32)
        dim = all_embeddings.shape[1]

        if len(passages) > 50000:
            quantizer = faiss.IndexFlatIP(dim)
            self._index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self._index.train(all_embeddings)
        else:
            self._index = faiss.IndexFlatIP(dim)

        self._index.add(all_embeddings)
        logger.info("Index FAISS construit : %d vecteurs de dimension %d", self._index.ntotal, dim)

        self._save_index()

    def _save_index(self) -> None:
        """Sauvegarde l'index et les métadonnées sur disque."""
        import faiss
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_dir / f"index_{self.lang}.faiss"))
        with open(self.index_dir / f"passage_ids_{self.lang}.pkl", "wb") as f:
            pickle.dump(self._passage_ids, f)
        with open(self.index_dir / f"passage_texts_{self.lang}.pkl", "wb") as f:
            pickle.dump(self._passage_texts, f)
        logger.info("Index sauvegardé → %s", self.index_dir)

    def load_index(self) -> None:
        """Charge l'index depuis le disque."""
        import faiss
        index_path = self.index_dir / f"index_{self.lang}.faiss"
        if not index_path.exists():
            raise FileNotFoundError(f"Index introuvable : {index_path}")

        self._index = faiss.read_index(str(index_path))
        with open(self.index_dir / f"passage_ids_{self.lang}.pkl", "rb") as f:
            self._passage_ids = pickle.load(f)
        with open(self.index_dir / f"passage_texts_{self.lang}.pkl", "rb") as f:
            self._passage_texts = pickle.load(f)
        logger.info("Index FAISS chargé : %d passages", len(self._passage_ids))

    def retrieve(self, query: str, k: int = 10) -> list[RetrievedPassage]:
        """
        Récupère les top-k passages pour une requête.

        Args:
            query: question en langage naturel
            k: nombre de passages à retourner

        Returns:
            Liste de passages classés par similarité décroissante
        """
        if self._index is None:
            self.load_index()
        if self._query_encoder is None:
            self._load_encoders()

        query_embedding = self._encode_texts([query], is_query=True).astype(np.float32)
        scores, indices = self._index.search(query_embedding, k)

        passages = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0:
                continue
            passages.append(RetrievedPassage(
                rank=rank,
                passage_id=self._passage_ids[idx],
                text=self._passage_texts[idx],
                dense_score=float(score),
            ))
        return passages

    def retrieve_batch(self, queries: list[str], k: int = 10) -> list[list[RetrievedPassage]]:
        """Récupère les top-k passages pour un lot de requêtes."""
        if self._index is None:
            self.load_index()
        if self._query_encoder is None:
            self._load_encoders()

        query_embeddings = self._encode_texts(queries, is_query=True).astype(np.float32)
        all_scores, all_indices = self._index.search(query_embeddings, k)

        results = []
        for scores, indices in zip(all_scores, all_indices):
            passages = []
            for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
                if idx < 0:
                    continue
                passages.append(RetrievedPassage(
                    rank=rank,
                    passage_id=self._passage_ids[idx],
                    text=self._passage_texts[idx],
                    dense_score=float(score),
                ))
            results.append(passages)
        return results

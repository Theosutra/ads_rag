__all__ = [
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    "ColBERTReranker",
    "CrossEncoderReranker",
    "get_reranker",
    "RetrievedPassage",
]


def __getattr__(name):
    if name in ("BM25Retriever", "RetrievedPassage"):
        from .bm25_retriever import BM25Retriever, RetrievedPassage
        return locals()[name]
    if name == "DenseRetriever":
        from .dense_retriever import DenseRetriever
        return DenseRetriever
    if name == "HybridRetriever":
        from .hybrid_retriever import HybridRetriever
        return HybridRetriever
    if name in ("ColBERTReranker", "CrossEncoderReranker", "get_reranker"):
        from .reranker import ColBERTReranker, CrossEncoderReranker, get_reranker
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

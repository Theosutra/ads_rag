__all__ = ["QAMetrics", "RetrievalMetrics", "RAGMetrics", "StatisticalTests"]


def __getattr__(name):
    if name == "QAMetrics":
        from .qa_metrics import QAMetrics
        return QAMetrics
    if name == "RetrievalMetrics":
        from .retrieval_metrics import RetrievalMetrics
        return RetrievalMetrics
    if name == "RAGMetrics":
        from .rag_metrics import RAGMetrics
        return RAGMetrics
    if name == "StatisticalTests":
        from .statistical_tests import StatisticalTests
        return StatisticalTests
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

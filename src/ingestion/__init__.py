__all__ = ["DatasetLoader", "TextNormalizer", "DocumentChunker"]


def __getattr__(name):
    if name == "DatasetLoader":
        from .loader import DatasetLoader
        return DatasetLoader
    if name == "TextNormalizer":
        from .normalizer import TextNormalizer
        return TextNormalizer
    if name == "DocumentChunker":
        from .chunker import DocumentChunker
        return DocumentChunker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

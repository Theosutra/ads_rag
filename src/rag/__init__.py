__all__ = ["ContextBuilder", "PromptFormatter", "RAGGenerator"]


def __getattr__(name):
    if name == "ContextBuilder":
        from .context_builder import ContextBuilder
        return ContextBuilder
    if name == "PromptFormatter":
        from .prompt_formatter import PromptFormatter
        return PromptFormatter
    if name == "RAGGenerator":
        from .generator import RAGGenerator
        return RAGGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

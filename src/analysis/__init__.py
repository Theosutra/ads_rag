__all__ = ["ErrorTaxonomy", "ErrorType", "ExperimentVisualizer"]


def __getattr__(name):
    if name in ("ErrorTaxonomy", "ErrorType"):
        from .error_taxonomy import ErrorTaxonomy, ErrorType
        return locals()[name]
    if name == "ExperimentVisualizer":
        from .visualizations import ExperimentVisualizer
        return ExperimentVisualizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

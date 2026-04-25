def __getattr__(name):
    if name == "LegalCaseClassifier":
        from .legal_case_classifier import LegalCaseClassifier
        return LegalCaseClassifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["LegalCaseClassifier"]

"""Kanoon scraper wrapper placed under src package.

Currently reuses functions from the legacy `Kanoon_cases.py` module at repo root
to avoid duplication. In a subsequent refactor, we can fully migrate the code here.
"""
try:
    from Kanoon_cases import (
        get_case_links,
        get_case_content,
        extract_judgment_section,
        get_case_summary,
        create_dataset,
    )
except ImportError as e:  # pragma: no cover
    raise

__all__ = [
    "get_case_links",
    "get_case_content",
    "extract_judgment_section",
    "get_case_summary",
    "create_dataset",
]

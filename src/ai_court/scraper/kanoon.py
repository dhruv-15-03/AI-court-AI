"""Kanoon scraper wrapper placed under src package.

Imports scraper functions from the legacy module within this package.
"""
from .legacy_kanoon import (
    get_case_links,
    get_case_content,
    extract_judgment_section,
    get_case_summary,
    create_dataset,
)

__all__ = [
    "get_case_links",
    "get_case_content",
    "extract_judgment_section",
    "get_case_summary",
    "create_dataset",
]

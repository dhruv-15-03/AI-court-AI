"""Pydantic schemas for the statute corpus."""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel


class StatuteSection(BaseModel):
    act_id: str
    act_name: str
    section_number: str
    heading: str
    body_text: str
    chapter: Optional[str] = None


class ActInfo(BaseModel):
    act_id: str
    full_name: str
    short_name: str
    year: int
    total_sections: int
    chapters: list[str] = []

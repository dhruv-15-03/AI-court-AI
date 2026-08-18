"""Phase 1 canonical schemas for ingestion & normalization.

These pydantic models define the intermediate and normalized representations
for Indian legal case ingestion and statute/section structures.
"""
from __future__ import annotations

from pydantic import BaseModel


class CaseRaw(BaseModel):
    case_id: str
    source_url: str
    court: str | None
    decision_date: str | None
    title: str | None
    full_text: str
    html: str | None
    scraped_at: str

class CaseSegment(BaseModel):
    case_id: str
    segment_id: str
    position: int
    role: str | None = None  # rhetorical role placeholder
    text: str

class Citation(BaseModel):
    case_id: str
    target_id: str
    raw_text: str
    offset: int

class StatuteReference(BaseModel):
    case_id: str
    section_id: str
    span_text: str
    offset: int

class SectionVersion(BaseModel):
    section_id: str
    act_id: str
    number: str
    version: int = 1
    heading: str | None
    body_text: str
    effective_start: str | None
    effective_end: str | None

class ActMetadata(BaseModel):
    act_id: str
    short_name: str
    long_name: str | None
    year: int | None
    version: int = 1

class NormalizedCase(BaseModel):
    case_id: str
    court: str | None
    decision_date: str | None
    title: str | None
    outcome_axis: str | None
    procedural_axis: str | None
    relief_axis: str | None
    raw_hash: str
    num_segments: int
    num_citations: int
    num_statutes: int
    created_at: str

__all__ = [
    'ActMetadata',
    'CaseRaw',
    'CaseSegment',
    'Citation',
    'NormalizedCase',
    'SectionVersion',
    'StatuteReference'
]

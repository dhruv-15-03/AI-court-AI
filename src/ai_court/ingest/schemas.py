"""Phase 1 canonical schemas for ingestion & normalization.

These pydantic models define the intermediate and normalized representations
for Indian legal case ingestion and statute/section structures.
"""
from __future__ import annotations
from pydantic import BaseModel
from typing import Optional

class CaseRaw(BaseModel):
    case_id: str
    source_url: str
    court: Optional[str]
    decision_date: Optional[str]
    title: Optional[str]
    full_text: str
    html: Optional[str]
    scraped_at: str

class CaseSegment(BaseModel):
    case_id: str
    segment_id: str
    position: int
    role: Optional[str] = None  # rhetorical role placeholder
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
    heading: Optional[str]
    body_text: str
    effective_start: Optional[str]
    effective_end: Optional[str]

class ActMetadata(BaseModel):
    act_id: str
    short_name: str
    long_name: Optional[str]
    year: Optional[int]
    version: int = 1

class NormalizedCase(BaseModel):
    case_id: str
    court: Optional[str]
    decision_date: Optional[str]
    title: Optional[str]
    outcome_axis: Optional[str]
    procedural_axis: Optional[str]
    relief_axis: Optional[str]
    raw_hash: str
    num_segments: int
    num_citations: int
    num_statutes: int
    created_at: str

__all__ = [
    'CaseRaw','CaseSegment','Citation','StatuteReference','SectionVersion','ActMetadata','NormalizedCase'
]

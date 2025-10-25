"""Phase 1 normalization pipeline.

Steps:
 1. Load raw judgment JSON blobs from store.
 2. Segment into paragraphs.
 3. Extract naive citations + statute references (regex placeholder).
 4. Produce parquet/JSON line outputs for downstream modeling.

This is intentionally lightweight; advanced enrichment phases will replace
heuristics with robust modules.
"""
from __future__ import annotations
import os, re, json, hashlib, time
from typing import Iterator, Dict, Any, List
from .schemas import CaseRaw, CaseSegment, Citation, StatuteReference, NormalizedCase
from bs4 import BeautifulSoup
from datetime import datetime

RAW_STORE_DIR = 'data/raw_html_store'
OUT_DIR = 'data/normalized'
os.makedirs(OUT_DIR, exist_ok=True)

CITATION_PATTERN = re.compile(r'(?i)(AIR\s?\d{4}|SCC\s?\d{4}|\b\d{4}\s*SCR)')
SECTION_PATTERN = re.compile(r'(?i)section\s+\d+[A-Za-z0-9]*')

BLOCK_TAGS = {"p","div","section","article","li"}

def html_to_text(html: str | None) -> str:
    if not html:
        return ""
    try:
        soup = BeautifulSoup(html, 'html.parser')
        # Remove scripts/styles
        for bad in soup(["script","style","noscript"]):
            bad.decompose()
        lines: List[str] = []
        for el in soup.find_all(BLOCK_TAGS):
            txt = el.get_text(" ", strip=True)
            if txt:
                lines.append(txt)
        if not lines:  # fallback to whole text
            return soup.get_text(" ", strip=True)
        # Collapse excessive whitespace
        out = re.sub(r"\s+"," ", "\n".join(lines)).strip()
        return out
    except Exception:
        # Fallback naive strip of tags
        return re.sub(r'<[^>]+>',' ', html)


def _iter_raw() -> Iterator[CaseRaw]:
    if not os.path.exists(RAW_STORE_DIR):
        return
    for fname in os.listdir(RAW_STORE_DIR):
        if not fname.endswith('.json'): continue
        path = os.path.join(RAW_STORE_DIR, fname)
        try:
            with open(path,'r',encoding='utf-8') as f:
                data = json.load(f)
            yield CaseRaw(**data)
        except Exception:
            continue

def segment_case(cr: CaseRaw) -> List[CaseSegment]:
    # Prefer HTML parsing if html present and full_text empty / short
    raw_text = cr.full_text
    if (not raw_text or len(raw_text) < 50) and getattr(cr,'html', None):
        raw_text = html_to_text(cr.html)  # type: ignore[arg-type]
    parts = [p.strip() for p in re.split(r'\n{2,}', raw_text or '') if p.strip()]
    segs: List[CaseSegment] = []
    for i, p in enumerate(parts):
        segs.append(CaseSegment(case_id=cr.case_id, segment_id=f"{cr.case_id}::seg::{i}", position=i, text=p))
    return segs

def extract_citations(text: str, case_id: str) -> List[Citation]:
    cites: List[Citation] = []
    for m in CITATION_PATTERN.finditer(text):
        cites.append(Citation(case_id=case_id, target_id='UNKNOWN', raw_text=m.group(0), offset=m.start()))
    return cites

def extract_sections(text: str, case_id: str) -> List[StatuteReference]:
    refs: List[StatuteReference] = []
    for m in SECTION_PATTERN.finditer(text):
        refs.append(StatuteReference(case_id=case_id, section_id='UNKNOWN', span_text=m.group(0), offset=m.start()))
    return refs


def run_normalize():
    os.makedirs(OUT_DIR, exist_ok=True)
    cases_out = open(os.path.join(OUT_DIR,'cases.jsonl'),'w',encoding='utf-8')
    segments_out = open(os.path.join(OUT_DIR,'segments.jsonl'),'w',encoding='utf-8')
    citations_out = open(os.path.join(OUT_DIR,'citations.jsonl'),'w',encoding='utf-8')
    statutes_out = open(os.path.join(OUT_DIR,'statute_refs.jsonl'),'w',encoding='utf-8')
    meta: List[NormalizedCase] = []
    count = 0
    try:
        for raw in _iter_raw():
            # Derive case_id if missing
            cid = raw.case_id if getattr(raw,'case_id', None) else hashlib.sha256(raw.source_url.encode('utf-8')).hexdigest()[:20]
            raw.case_id = cid
            segs = segment_case(raw)
            all_text = '\n'.join(s.text for s in segs)
            cites: List[Citation] = []
            sections: List[StatuteReference] = []
            for s in segs:
                cites.extend(extract_citations(s.text, cid))
                sections.extend(extract_sections(s.text, cid))
            material_text = raw.full_text or raw.html or ''
            raw_hash = hashlib.sha256(material_text.encode('utf-8')).hexdigest()
            norm = NormalizedCase(
                case_id=cid,
                court=raw.court,
                decision_date=raw.decision_date,
                title=raw.title,
                outcome_axis=None,
                procedural_axis=None,
                relief_axis=None,
                raw_hash=raw_hash,
                num_segments=len(segs),
                num_citations=len(cites),
                num_statutes=len(sections),
                created_at=datetime.utcnow().isoformat()+'Z'
            )
            cases_out.write(raw.model_dump_json()+"\n")
            for s in segs:
                segments_out.write(s.model_dump_json()+"\n")
            for c in cites:
                citations_out.write(c.model_dump_json()+"\n")
            for st in sections:
                statutes_out.write(st.model_dump_json()+"\n")
            meta.append(norm)
            count += 1
    finally:
        cases_out.close(); segments_out.close(); citations_out.close(); statutes_out.close()
    # Write manifest
    manifest = {
        'cases': count,
        'generated_at': datetime.utcnow().isoformat()+'Z',
        'totals': {
            'segments': sum(m.num_segments for m in meta),
            'citations': sum(m.num_citations for m in meta),
            'statute_refs': sum(m.num_statutes for m in meta)
        },
        'hash': hashlib.sha256('\n'.join(sorted(m.raw_hash for m in meta)).encode('utf-8')).hexdigest()
    }
    with open(os.path.join(OUT_DIR,'manifest.json'),'w',encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    print(f"[normalize] Processed {count} cases -> manifest hash {manifest['hash']}")

if __name__ == '__main__':
    run_normalize()

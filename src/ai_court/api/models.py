from typing import Optional, Dict, List
from pydantic import BaseModel, Field

class AnalyzeRequest(BaseModel):
    case_type: Optional[str] = Field(default="Unknown", max_length=100)
    summary: Optional[str] = Field(default=None, max_length=5000)
    parties: Optional[str] = Field(default=None, max_length=2000)
    violence_level: Optional[str] = Field(default=None, max_length=100)
    weapon: Optional[str] = Field(default=None, max_length=50)
    police_report: Optional[str] = Field(default=None, max_length=50)
    witnesses: Optional[str] = Field(default=None, max_length=50)
    premeditation: Optional[str] = Field(default=None, max_length=50)
    employment_duration: Optional[str] = Field(default=None, max_length=100)
    children: Optional[str] = Field(default=None, max_length=50)
    marriage_duration: Optional[str] = Field(default=None, max_length=100)
    dispute_type: Optional[str] = Field(default=None, max_length=100)
    document_evidence: Optional[str] = Field(default=None, max_length=50)
    monetary_value: Optional[str] = Field(default=None, max_length=100)
    prior_relationship: Optional[str] = Field(default=None, max_length=200)
    attempts_resolution: Optional[str] = Field(default=None, max_length=50)

    def combined_text(self) -> str:
        # This legacy method only includes typed fields. We prefer synthesizing from raw.
        parts: List[str] = []
        data = self.model_dump()
        ct = data.get("case_type", "") or ""
        # Put summary first if provided
        if data.get("summary"):
            parts.append(str(data.get("summary")))
        for k, v in data.items():
            if k in ("case_type", "summary"):
                continue
            if v:
                parts.append(f"{k.replace('_',' ')}: {v}")
        return (f"{ct} " + ". ".join(parts)).strip()


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=5000)
    k: int = Field(default=5, ge=1, le=20)


class DriftCompareRequest(BaseModel):
    counts: Dict[str, int]
    minimum_total: int = Field(default=1, ge=1)

    def normalized(self, classes: List[str]) -> List[float]:
        total = sum(v for v in self.counts.values() if isinstance(v, (int, float)))
        if total <= 0:
            return [0.0 for _ in classes]
        return [float(self.counts.get(c, 0)) / total for c in classes]


from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    case_type: str | None = Field(default="Unknown", max_length=100)
    summary: str | None = Field(default=None, max_length=5000)
    parties: str | None = Field(default=None, max_length=2000)
    violence_level: str | None = Field(default=None, max_length=100)
    weapon: str | None = Field(default=None, max_length=50)
    police_report: str | None = Field(default=None, max_length=50)
    witnesses: str | None = Field(default=None, max_length=50)
    premeditation: str | None = Field(default=None, max_length=50)
    employment_duration: str | None = Field(default=None, max_length=100)
    children: str | None = Field(default=None, max_length=50)
    marriage_duration: str | None = Field(default=None, max_length=100)
    dispute_type: str | None = Field(default=None, max_length=100)
    document_evidence: str | None = Field(default=None, max_length=50)
    monetary_value: str | None = Field(default=None, max_length=100)
    prior_relationship: str | None = Field(default=None, max_length=200)
    attempts_resolution: str | None = Field(default=None, max_length=50)

    def combined_text(self) -> str:
        # This legacy method only includes typed fields. We prefer synthesizing from raw.
        parts: list[str] = []
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
    counts: dict[str, int]
    minimum_total: int = Field(default=1, ge=1)

    def normalized(self, classes: list[str]) -> list[float]:
        total = sum(v for v in self.counts.values() if isinstance(v, (int, float)))
        if total <= 0:
            return [0.0 for _ in classes]
        return [float(self.counts.get(c, 0)) / total for c in classes]

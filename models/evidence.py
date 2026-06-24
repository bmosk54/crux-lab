"""Pydantic models for structured evidence and paper strength scoring."""

from __future__ import annotations

import math
from datetime import date
from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field

EvidenceType = Literal[
    "rct",           # Randomised controlled trial
    "meta_analysis", # Systematic review or meta-analysis
    "cohort",        # Cohort, case-control, or large observational study
    "animal_model",  # In vivo animal experiment
    "in_vitro",      # Cell culture or ex vivo experiment
    "mechanistic",   # Proposed mechanism / theoretical (no direct experimental test)
    "case_report",   # Case report or case series
]

Directness = Literal[
    "direct",     # Study explicitly tested or measured this claim
    "indirect",   # Finding implies the claim but didn't directly test it
    "tangential", # Related but requires reasoning steps to connect to hypothesis
]


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str = Field(description="A specific claim extracted from the abstract")
    evidence_type: EvidenceType = Field(
        description="The methodological category of the study producing this claim"
    )
    directness: Directness = Field(
        description="How directly this claim addresses the hypothesis"
    )


class EvidencePaper(BaseModel):
    title: str
    authors: str = ""
    journal: str | None = None
    published: str | None = None  # YYYY-MM-DD
    pmid: str | None = None
    pmcid: str | None = None
    doi: str | None = None
    cited_by_count: int = 0
    abstract_snippet: str = ""
    supporting_claims: List[Claim] = Field(default_factory=list)
    refuting_claims: List[Claim] = Field(default_factory=list)
    paper_strength: float = Field(default=0.0, ge=0.0, le=1.0)


class EvidenceSet(BaseModel):
    hypothesis: str
    supporting_papers: List[EvidencePaper] = Field(default_factory=list)
    refuting_papers: List[EvidencePaper] = Field(default_factory=list)


def compute_paper_strength(cited_by_count: int, published: str | None) -> float:
    """
    Score a paper 0–1 based on citation rate adjusted for publication age.

    Uses a log scale where ~30 citations/year maps to strength ≈ 1.0.
    Older papers get a small longevity bonus (up to +0.1) for sustained impact.
    """
    current_year = date.today().year
    pub_year = current_year
    if published:
        try:
            pub_year = int(published[:4])
        except (ValueError, IndexError):
            pass

    years_active = max(1, current_year - pub_year + 1)
    citation_rate = cited_by_count / years_active

    # log scale: 30 cit/year → 1.0
    base = min(1.0, math.log1p(citation_rate) / math.log1p(30))

    # small longevity bonus: papers active ≥10 years that are still cited
    longevity_bonus = 0.1 * min(1.0, years_active / 10) if cited_by_count > 0 else 0.0

    return round(min(1.0, base + longevity_bonus), 3)

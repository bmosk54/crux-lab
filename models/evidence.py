"""Pydantic models for structured evidence, candidate ranking, and scoring."""

from __future__ import annotations

import math
import re
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
    "direct",     # Study directly measured the exact hypothesis variable(s)
    "indirect",   # Measured something one causal step from the hypothesis variable
    "tangential", # Connection requires multiple reasoning steps
]

# ---------------------------------------------------------------------------
# Ranking weights — relevance matters most, then citation impact, then
# how specifically the paper targets the hypothesis. Must sum to 1.0.
# ---------------------------------------------------------------------------
W_RELEVANCE = 0.45
W_CITATION = 0.30
W_SPECIFICITY = 0.25

# Relevance floor for papers that entered the pool only via a citation-sorted
# (high-impact) pass: they matched the query terms but lack a relevance rank,
# so give them a moderate relevance rather than zero.
RELEVANCE_FLOOR_ESTABLISHED = 0.4

_STOPWORDS = {
    "the", "and", "for", "are", "was", "with", "that", "this", "from", "can",
    "has", "have", "had", "but", "not", "all", "any", "may", "via", "its",
    "into", "than", "then", "out", "over", "under", "more", "less", "such",
    "due", "per", "use", "used", "using", "between", "during", "within",
    "associated", "role", "effect", "effects", "study", "studies", "based",
}


# ---------------------------------------------------------------------------
# Final evidence models (LLM output)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Candidate model (Phase-1 discovery + Python ranking)
# ---------------------------------------------------------------------------


class CandidatePaper(BaseModel):
    source: str
    ext_id: str
    title: str = ""
    authors: str = ""
    journal: str | None = None
    published: str | None = None
    pmid: str | None = None
    pmcid: str | None = None
    doi: str | None = None
    cited_by_count: int = 0
    abstract: str = ""

    # Ranking signals
    pools: List[str] = Field(default_factory=list)  # "supporting" / "refuting"
    found_count: int = 0
    best_rank: float = 0.0
    established: bool = False  # surfaced by a citation-sorted (high-impact) pass
    relevance: float = 0.0
    citation_score: float = 0.0
    specificity: float = 0.0
    composite: float = 0.0


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------


def extract_keywords(text: str) -> set[str]:
    """Tokenize text into content keywords (lowercased, stopwords removed)."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in tokens if len(t) > 2 and t not in _STOPWORDS}


def compute_specificity(title: str, hypothesis_keywords: set[str]) -> float:
    """Fraction of hypothesis keywords present in the paper title (0–1).

    Rewards papers whose title narrowly targets the hypothesis variables over
    broad reviews that merely mention the topic.
    """
    if not hypothesis_keywords:
        return 0.0
    title_keywords = extract_keywords(title)
    matched = hypothesis_keywords & title_keywords
    return round(len(matched) / len(hypothesis_keywords), 3)


def compute_paper_strength(cited_by_count: int, published: str | None) -> float:
    """Score a paper 0–1 by citation rate adjusted for publication age.

    Log scale where ~30 citations/year maps to ~1.0, plus a small longevity
    bonus for papers that have stayed cited over many years.
    """
    current_year = date.today().year
    pub_year = current_year
    if published:
        try:
            pub_year = int(published[:4])
        except (ValueError, IndexError):
            pass

    years_active = max(1, current_year - pub_year + 1)
    citation_rate = (cited_by_count or 0) / years_active

    base = min(1.0, math.log1p(citation_rate) / math.log1p(30))
    longevity_bonus = 0.1 * min(1.0, years_active / 10) if cited_by_count else 0.0
    return round(min(1.0, base + longevity_bonus), 3)


def compute_composite(relevance: float, citation: float, specificity: float) -> float:
    """Weighted blend of the three ranking signals (relevance weighted highest)."""
    score = (
        W_RELEVANCE * relevance
        + W_CITATION * citation
        + W_SPECIFICITY * specificity
    )
    return round(score, 4)

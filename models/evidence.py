"""Pydantic models + helpers for the crux-lab pipeline.

Data contract across stages:
  Stage 0 (decompose)      -> Decomposition           (shared with synthesizer)
  Stage 2-3 (search/rank)  -> CandidatePaper          (title-level)
  Stage 3.x (abstract pass)-> EvidencePaper           (axes + key_findings)
  Stage 3.5 (full text)    -> EvidencePaper.fulltext_sections
  Stage 4 (synthesis)      -> Verdict                 (consumes Decomposition + EvidenceSet)
"""

from __future__ import annotations

import math
import re
from datetime import date
from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Shared enums
# ---------------------------------------------------------------------------

EvidenceType = Literal[
    "rct",           # Randomised controlled trial
    "meta_analysis", # Systematic review or meta-analysis
    "cohort",        # Cohort, case-control, or large observational study
    "animal_model",  # In vivo animal experiment
    "in_vitro",      # Cell culture or ex vivo experiment
    "mechanistic",   # Proposed mechanism / theoretical (no direct experimental test)
    "case_report",   # Case report or case series
    "review",        # Narrative review / opinion
]

Directness = Literal[
    "direct",     # Tests the claim's actual variables
    "indirect",   # Tests a proxy / one causal step away
    "tangential", # Connection requires multiple reasoning steps
]

Stance = Literal["support", "refute", "mixed"]

VenueQuality = Literal["high", "ok", "low"]

ClaimType = Literal[
    "causal", "functional", "mechanistic", "therapeutic",
    "association", "paradigm", "absence", "comparative",
]

SubClaimLabel = Literal["supported", "refuted", "mixed", "untested"]

_STOPWORDS = {
    "the", "and", "for", "are", "was", "with", "that", "this", "from", "can",
    "has", "have", "had", "but", "not", "all", "any", "may", "via", "its",
    "into", "than", "then", "out", "over", "under", "more", "less", "such",
    "due", "per", "use", "used", "using", "between", "during", "within",
    "associated", "role", "effect", "effects", "study", "studies", "based",
    "will", "are", "is", "of", "in", "on", "to", "a", "an",
}


# ---------------------------------------------------------------------------
# Stage 0 — decomposition (shared contract with the synthesizer)
# ---------------------------------------------------------------------------


class Decomposition(BaseModel):
    """Structured breakdown of the input claim. Every downstream stage reads this."""

    claim: str = ""  # original claim verbatim (filled by the pipeline, not the LLM)
    sub_claims: List[str] = Field(default_factory=list)
    load_bearing_modifiers: List[str] = Field(default_factory=list)
    claim_type: ClaimType = "causal"
    established_knowledge: str = ""  # consensus position + RELATION (extends/contradicts/…)
    named_entities: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Candidate (title-level discovery)
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
    is_open_access: bool = False
    abstract: str = ""

    # Discovery provenance (informational only — no weighted formula)
    pools: List[str] = Field(default_factory=list)  # supporting / refuting / entity
    found_count: int = 0
    established: bool = False
    specificity: float = 0.0


# ---------------------------------------------------------------------------
# Evidence (abstract pass + full text) — the working-set unit
# ---------------------------------------------------------------------------


class EvidencePaper(BaseModel):
    title: str
    authors: str = ""
    journal: str | None = None
    published: str | None = None
    pmid: str | None = None
    pmcid: str | None = None
    doi: str | None = None
    cited_by_count: int = 0

    # Separately-tracked axes (Change 3) — never collapsed into one score
    stance: Stance = "mixed"
    directness: Directness = "indirect"
    evidence_tier: EvidenceType = "review"
    entity_specific: bool = False
    venue_quality: VenueQuality = "ok"
    is_consensus_paper: bool = False
    key_findings: List[str] = Field(default_factory=list)

    # Provenance / depth
    abstract_snippet: str = ""
    fulltext_sections: str = ""   # extracted Results/Discussion (pivotal papers only)
    abstract_only: bool = True    # False once full text is attached
    reserved_as: str | None = None  # which reserved slot guaranteed it (debug/provenance)
    paper_strength: float = Field(default=0.0, ge=0.0, le=1.0)


class EvidenceSet(BaseModel):
    hypothesis: str
    papers: List[EvidencePaper] = Field(default_factory=list)

    def by_stance(self, stance: Stance) -> List[EvidencePaper]:
        return [p for p in self.papers if p.stance == stance]


# ---------------------------------------------------------------------------
# Stage 4 — verdict (the merged synthesis+verdict output)
# ---------------------------------------------------------------------------


class SubClaimVerdict(BaseModel):
    sub_claim: str
    label: SubClaimLabel = "untested"
    confidence: int = Field(default=0, ge=0, le=100)
    basis: str = ""


class Steelman(BaseModel):
    statement: str = ""
    confidence: int = Field(default=0, ge=0, le=100)


class Verdict(BaseModel):
    # Overall claim AS WRITTEN
    support_score: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="0.0 = strongly refuted, 0.5 = mixed, 1.0 = strongly supported",
    )
    overall_label: SubClaimLabel = "untested"
    overall_confidence: int = Field(default=0, ge=0, le=100)

    sub_claims: List[SubClaimVerdict] = Field(default_factory=list)
    consensus_positioning: str = ""
    counter_evidence_assessment: str = ""
    steelman: Steelman = Field(default_factory=Steelman)
    decisive_experiment: str = ""
    untested_components: List[str] = Field(default_factory=list)
    reasoning: str = ""

    # Set by the pipeline when claim_type == "absence" (null search ≠ confirmation)
    absence_coverage_note: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_keywords(text: str) -> set[str]:
    """Tokenize text into content keywords (lowercased, stopwords removed)."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in tokens if len(t) > 2 and t not in _STOPWORDS}


def compute_specificity(title: str, hypothesis_keywords: set[str]) -> float:
    """Fraction of hypothesis keywords present in the paper title (0–1).

    Used only as a cheap minimum-relevance gate — not as a weighted ranking score.
    """
    if not hypothesis_keywords:
        return 0.0
    title_keywords = extract_keywords(title)
    matched = hypothesis_keywords & title_keywords
    return round(len(matched) / len(hypothesis_keywords), 3)


def compute_paper_strength(cited_by_count: int, published: str | None) -> float:
    """Score a paper 0–1 by citation rate adjusted for publication age."""
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


def citations_per_year(cited_by_count: int, published: str | None) -> float:
    """Impact tiebreaker; returns 0 for papers < 2 years old (too new to judge)."""
    current_year = date.today().year
    pub_year = current_year
    if published:
        try:
            pub_year = int(published[:4])
        except (ValueError, IndexError):
            pass
    age = current_year - pub_year
    if age < 2:
        return 0.0
    return round((cited_by_count or 0) / max(1, age), 2)

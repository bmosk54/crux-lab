"""Evidence gatherer: a Python-orchestrated pipeline over a single Claude model.

Pipeline stages:
  1. generate_queries   (LLM)    — frame 3 supporting + 3 refuting search queries
  2. collect_candidates (Python) — dual-pass search + minimum-relevance gate
  3. screen_candidates  (LLM)    — judge relevance & quality, pick top ~10 per side
  4. attach_abstracts   (Python) — fetch abstracts for the finalists by ID
  5. extract_evidence   (LLM)    — read abstracts, extract & classify claims

Ranking uses the model's judgement (relevance first, then quality signals like
citations/recency/specificity) rather than a hardcoded weighted formula. Python
only applies a light minimum-relevance gate to drop obvious junk before screening.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agents.llm import complete, parse_tagged_json
from models.evidence import (
    CandidatePaper,
    EvidenceSet,
    compute_paper_strength,
    compute_specificity,
    extract_keywords,
)
from tools.europepmc import fetch_abstracts, run_search

RELEVANCE_PAGE_SIZE = 20
ESTABLISHED_PAGE_SIZE = 10
ESTABLISHED_MIN_CITATIONS = 20
TOP_PER_SIDE = 10
ABSTRACT_CHAR_LIMIT = 1600


@dataclass
class GatherResult:
    """Everything produced by the gathering pipeline (for output + debugging)."""

    hypothesis: str
    supporting_queries: list[str] = field(default_factory=list)
    refuting_queries: list[str] = field(default_factory=list)
    supporting_candidates: list[CandidatePaper] = field(default_factory=list)
    refuting_candidates: list[CandidatePaper] = field(default_factory=list)
    evidence: EvidenceSet | None = None


# ---------------------------------------------------------------------------
# Stage 1 — query generation
# ---------------------------------------------------------------------------

QUERY_GEN_SYSTEM = """
You generate Europe PMC search queries to gather evidence about a scientific
hypothesis. Produce two sets:

- supporting_queries: 3 queries likely to surface evidence the hypothesis is TRUE
  (positive findings, supportive mechanisms, successful interventions).
- refuting_queries: 3 queries likely to surface evidence it is FALSE or limited
  (null results, opposing mechanisms, failed replications, limitations, controversy).

Guidelines:
- Use precise scientific terminology and key variables from the hypothesis.
- You may use Europe PMC field syntax (e.g. "exact phrase", MESH:Term).
- Make the 3 queries within each set DIFFERENT from each other (vary angle,
  mechanism, population, or study type).
- At least one query per set should target recent/novel work.

Output EXACTLY this and nothing else:

<queries>
{"supporting_queries": ["q1", "q2", "q3"], "refuting_queries": ["q1", "q2", "q3"]}
</queries>
""".strip()


def _fallback_queries(hypothesis: str) -> tuple[list[str], list[str]]:
    return (
        [
            hypothesis,
            f"{hypothesis} mechanism evidence",
            f"{hypothesis} recent findings",
        ],
        [
            f"{hypothesis} limitations",
            f"{hypothesis} no effect OR null result",
            f"{hypothesis} contradictory OR controversy",
        ],
    )


async def generate_queries(hypothesis: str, *, model: str = "haiku") -> tuple[list[str], list[str]]:
    text = await complete(QUERY_GEN_SYSTEM, f"Hypothesis:\n\n{hypothesis}", model=model)
    data = parse_tagged_json(text, "queries")
    if isinstance(data, dict):
        sup = [q for q in data.get("supporting_queries", []) if isinstance(q, str) and q.strip()]
        ref = [q for q in data.get("refuting_queries", []) if isinstance(q, str) and q.strip()]
        if sup and ref:
            return sup, ref
    return _fallback_queries(hypothesis)


# ---------------------------------------------------------------------------
# Stage 2 — discovery search + minimum-relevance gate
# ---------------------------------------------------------------------------


async def collect_candidates(
    hypothesis: str,
    supporting_queries: list[str],
    refuting_queries: list[str],
) -> list[CandidatePaper]:
    """Run dual-pass discovery searches, dedupe, and apply a minimum-relevance gate.

    No weighted ranking happens here — the model screens for relevance next.
    """
    keywords = extract_keywords(hypothesis)
    candidates: dict[tuple[str, str], CandidatePaper] = {}

    def register(hit: dict, side: str, established: bool) -> None:
        source, ext_id = hit.get("source"), hit.get("id")
        if not source or not ext_id:
            return
        key = (source, ext_id)
        cand = candidates.get(key)
        if cand is None:
            cand = CandidatePaper(
                source=source,
                ext_id=ext_id,
                title=hit.get("title") or "",
                authors=hit.get("authors") or "",
                journal=hit.get("journal"),
                published=hit.get("published"),
                pmid=hit.get("pmid"),
                pmcid=hit.get("pmcid"),
                doi=hit.get("doi"),
                cited_by_count=int(hit.get("cited_by_count") or 0),
            )
            candidates[key] = cand
        cand.found_count += 1
        cand.established = cand.established or established
        if side not in cand.pools:
            cand.pools.append(side)

    async def process(queries: list[str], side: str) -> None:
        for q in queries:
            # Pass A: relevance-sorted — best topical matches (including recent)
            rel = await run_search(q, page_size=RELEVANCE_PAGE_SIZE, sort="relevance")
            for hit in rel.get("results", []):
                register(hit, side, established=False)
            # Pass B: citation-sorted — guarantees established high-impact papers
            # enter the pool even when relevance ordering buried them
            est = await run_search(
                q,
                page_size=ESTABLISHED_PAGE_SIZE,
                sort="citedByCount",
                min_citations=ESTABLISHED_MIN_CITATIONS,
            )
            for hit in est.get("results", []):
                register(hit, side, established=True)

    await process(supporting_queries, "supporting")
    await process(refuting_queries, "refuting")

    # Minimum-relevance gate: keep a paper only if its title shares at least one
    # keyword with the hypothesis, OR it is an established high-impact paper that
    # earned its place via citations (its title may use different wording).
    gated: list[CandidatePaper] = []
    for cand in candidates.values():
        cand.specificity = compute_specificity(cand.title, keywords)
        if cand.specificity > 0 or cand.established:
            gated.append(cand)
    return gated


# ---------------------------------------------------------------------------
# Stage 3 — screen candidates with judgement (replaces weighted ranking)
# ---------------------------------------------------------------------------

SCREEN_SYSTEM = """
You are screening candidate papers (titles only, no abstracts) to decide which are
worth deep-reading as evidence about a hypothesis. You receive a numbered list with
each paper's title, publication year, citation count, and which search surfaced it.

Select the papers most worth reading:
- up to 10 as SUPPORTING candidates (titles suggesting evidence the hypothesis is true)
- up to 10 as REFUTING candidates (titles suggesting evidence against or limiting it)

Use judgement, NOT a fixed formula:
- RELEVANCE comes first. Does the title indicate the paper actually addresses the
  hypothesis's specific variables? Exclude papers that are only loosely related
  (reviews of adjacent topics, a different disease/organism with no clear link,
  pure methods/assay papers). It is fine to exclude a highly-cited paper if it
  isn't relevant, and fine to return fewer than 10 per side.
- Then weigh quality signals together: citation impact, recency (new findings),
  and how specific/empirical the title sounds (primary studies, trials, direct
  measurements over broad narrative reviews).
- Prefer a DIVERSE set of study types and angles over near-duplicate titles.

A paper may appear in BOTH lists if its title suggests mixed or broadly relevant
findings.

Output EXACTLY this and nothing else:

<selection>
{"supporting": [paper numbers], "refuting": [paper numbers]}
</selection>
""".strip()


def _format_screen_line(index: int, cand: CandidatePaper) -> str:
    surfaced = "/".join(cand.pools) or "?"
    year = (cand.published or "?")[:4]
    return f"{index}. [{cand.cited_by_count} cit · {year} · {surfaced}] {cand.title}"


def _coerce_indices(values: object, n: int) -> list[int]:
    out: list[int] = []
    if isinstance(values, list):
        for v in values:
            try:
                i = int(v)
            except (TypeError, ValueError):
                continue
            if 1 <= i <= n and i not in out:
                out.append(i)
    return out


async def screen_candidates(
    hypothesis: str,
    candidates: list[CandidatePaper],
    top_per_side: int = TOP_PER_SIDE,
    *,
    model: str = "sonnet",
) -> tuple[list[CandidatePaper], list[CandidatePaper]]:
    if not candidates:
        return [], []

    ordered = sorted(candidates, key=lambda c: c.cited_by_count, reverse=True)
    listing = "\n".join(_format_screen_line(i, c) for i, c in enumerate(ordered, 1))
    prompt = f"Hypothesis:\n{hypothesis}\n\nCandidate papers ({len(ordered)}):\n{listing}"

    text = await complete(SCREEN_SYSTEM, prompt, model=model)
    data = parse_tagged_json(text, "selection") or {}
    sup_idx = _coerce_indices(data.get("supporting"), len(ordered))[:top_per_side]
    ref_idx = _coerce_indices(data.get("refuting"), len(ordered))[:top_per_side]

    if not sup_idx and not ref_idx:
        # Fallback: keep the most-cited from each original search pool
        sup_idx = [i for i, c in enumerate(ordered, 1) if "supporting" in c.pools][:top_per_side]
        ref_idx = [i for i, c in enumerate(ordered, 1) if "refuting" in c.pools][:top_per_side]

    supporting = [ordered[i - 1] for i in sup_idx]
    refuting = [ordered[i - 1] for i in ref_idx]

    # Re-label pools to reflect the screening decision (used for the "retrieved as"
    # hint downstream); a paper selected for both sides keeps both labels.
    sup_keys = {(c.source, c.ext_id) for c in supporting}
    ref_keys = {(c.source, c.ext_id) for c in refuting}
    for cand in {(c.source, c.ext_id): c for c in [*supporting, *refuting]}.values():
        key = (cand.source, cand.ext_id)
        cand.pools = []
        if key in sup_keys:
            cand.pools.append("supporting")
        if key in ref_keys:
            cand.pools.append("refuting")

    return supporting, refuting


# ---------------------------------------------------------------------------
# Stage 4 — fetch abstracts for the finalists
# ---------------------------------------------------------------------------


async def attach_abstracts(
    supporting: list[CandidatePaper], refuting: list[CandidatePaper]
) -> list[CandidatePaper]:
    """Fetch abstracts for the unique union of finalists and attach them."""
    union: dict[tuple[str, str], CandidatePaper] = {}
    for cand in [*supporting, *refuting]:
        union[(cand.source, cand.ext_id)] = cand

    fetched = await fetch_abstracts(list(union.keys()))
    for (_source, ext_id), cand in union.items():
        record = fetched.get(ext_id)
        if record and record.get("abstract"):
            cand.abstract = record["abstract"]
    return list(union.values())


# ---------------------------------------------------------------------------
# Stage 5 — extract evidence
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM = """
You are a rigorous scientific evidence analyst. You are given a hypothesis and a
set of candidate papers (already screened for relevance) with their abstracts.

Your job: read each abstract, extract specific claims, classify them, and select
the final strongest papers on each side.

## Extract claims

For each paper, extract:
- supporting_claims: specific findings that AGREE with the hypothesis
- refuting_claims:   specific findings that DISAGREE with or limit it
A paper may contribute both. Quote/paraphrase concrete findings, not vague themes.

## Classify each claim

evidence_type — the methodological category of the study:
  "rct"          — Randomised controlled trial
  "meta_analysis"— Systematic review or meta-analysis
  "cohort"       — Cohort, case-control, or large observational study
  "animal_model" — In vivo animal experiment
  "in_vitro"     — Cell culture or ex vivo experiment
  "mechanistic"  — Proposed mechanism or theoretical framework (no direct test)
  "case_report"  — Case report or case series

directness — does the abstract's PRIMARY measurement match the hypothesis variable:
  "direct"    — The paper directly measured the EXACT variable(s) in the hypothesis
                (regardless of the study's original purpose)
  "indirect"  — The paper measured something one causal step away, needing one
                inferential step to connect to the hypothesis
  "tangential"— Connection needs multiple reasoning steps, or context differs
                substantially (different organism, disease vs. aging, etc.)

## Select the final set

- Choose the 3-5 strongest SUPPORTING and 3-5 strongest REFUTING papers.
- Favour DIVERSITY of study design over multiple papers making the same point.
- Prefer papers with direct measurements and stronger evidence types.
- Copy each paper's metadata (title, authors, journal, published, ids,
  cited_by_count) EXACTLY as provided — do not invent or alter values.

## Output format

Output EXACTLY this and nothing after the closing tag:

<evidence>
{
  "hypothesis": "<hypothesis exactly as given>",
  "supporting_papers": [
    {
      "title": "...", "authors": "...", "journal": "...", "published": "YYYY-MM-DD",
      "pmid": "..." or null, "pmcid": "..." or null, "doi": "..." or null,
      "cited_by_count": 0,
      "abstract_snippet": "<most relevant 1-3 sentences from the abstract>",
      "supporting_claims": [{"claim": "...", "evidence_type": "rct", "directness": "direct"}],
      "refuting_claims": [{"claim": "...", "evidence_type": "cohort", "directness": "indirect"}]
    }
  ],
  "refuting_papers": [ ... same schema ... ]
}
</evidence>
""".strip()


def _format_candidate_block(index: int, cand: CandidatePaper) -> str:
    abstract = (cand.abstract or "").strip()
    if len(abstract) > ABSTRACT_CHAR_LIMIT:
        abstract = abstract[:ABSTRACT_CHAR_LIMIT] + "…"
    retrieved_as = "/".join(cand.pools) or "unknown"
    return (
        f"[Paper {index}] (retrieved as: {retrieved_as})\n"
        f"title: {cand.title}\n"
        f"authors: {cand.authors}\n"
        f"journal: {cand.journal or 'n/a'}\n"
        f"published: {cand.published or 'n/a'}\n"
        f"pmid: {cand.pmid or 'null'} | pmcid: {cand.pmcid or 'null'} | doi: {cand.doi or 'null'}\n"
        f"cited_by_count: {cand.cited_by_count}\n"
        f"abstract: {abstract or '(no abstract available)'}\n"
    )


async def extract_evidence(
    hypothesis: str, papers: list[CandidatePaper], *, model: str = "sonnet"
) -> EvidenceSet:
    papers_with_abstracts = [p for p in papers if p.abstract.strip()]
    if not papers_with_abstracts:
        return EvidenceSet(hypothesis=hypothesis)

    blocks = "\n".join(
        _format_candidate_block(i, p) for i, p in enumerate(papers_with_abstracts, 1)
    )
    prompt = (
        f"Hypothesis:\n{hypothesis}\n\n"
        f"Candidate papers ({len(papers_with_abstracts)}):\n\n{blocks}"
    )
    text = await complete(EXTRACTION_SYSTEM, prompt, model=model)
    evidence = _parse_evidence(text, hypothesis)
    _refresh_metadata(evidence, papers_with_abstracts)
    return evidence


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


async def gather_evidence(
    hypothesis: str,
    *,
    verbose: bool = True,
    model_query_gen: str = "haiku",
    model_screening: str = "sonnet",
    model_extraction: str = "sonnet",
) -> GatherResult:
    def log(msg: str) -> None:
        if verbose:
            print(msg)

    log(f"  → generating search queries (model: {model_query_gen})...")
    supporting_queries, refuting_queries = await generate_queries(
        hypothesis, model=model_query_gen
    )

    n_queries = len(supporting_queries) + len(refuting_queries)
    log(f"  → running {n_queries * 2} discovery searches ({n_queries} queries × relevance+citation passes)...")
    candidates = await collect_candidates(hypothesis, supporting_queries, refuting_queries)
    log(f"  → {len(candidates)} candidates passed the relevance gate")

    log(f"  → screening candidates (model: {model_screening})...")
    supporting, refuting = await screen_candidates(
        hypothesis, candidates, model=model_screening
    )
    log(f"  → selected {len(supporting)} supporting, {len(refuting)} refuting")

    papers = await attach_abstracts(supporting, refuting)
    log(f"  → fetched abstracts for {sum(1 for p in papers if p.abstract.strip())} papers")

    log(f"  → extracting evidence (model: {model_extraction})...")
    evidence = await extract_evidence(hypothesis, papers, model=model_extraction)

    return GatherResult(
        hypothesis=hypothesis,
        supporting_queries=supporting_queries,
        refuting_queries=refuting_queries,
        supporting_candidates=supporting,
        refuting_candidates=refuting,
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_evidence(text: str, hypothesis: str) -> EvidenceSet:
    data = parse_tagged_json(text, "evidence")
    if isinstance(data, dict):
        try:
            return EvidenceSet(**data)
        except (TypeError, ValueError) as exc:
            print(f"[warning] Could not parse evidence JSON: {exc}")
    return EvidenceSet(hypothesis=hypothesis)


def _refresh_metadata(evidence: EvidenceSet, candidates: list[CandidatePaper]) -> None:
    """Overwrite LLM-reported citation/date with authoritative candidate data,
    then recompute paper_strength deterministically."""
    by_doi = {c.doi.lower(): c for c in candidates if c.doi}
    by_pmid = {c.pmid: c for c in candidates if c.pmid}
    by_title = {c.title.lower(): c for c in candidates if c.title}

    for paper in [*evidence.supporting_papers, *evidence.refuting_papers]:
        match = None
        if paper.doi and paper.doi.lower() in by_doi:
            match = by_doi[paper.doi.lower()]
        elif paper.pmid and paper.pmid in by_pmid:
            match = by_pmid[paper.pmid]
        elif paper.title and paper.title.lower() in by_title:
            match = by_title[paper.title.lower()]

        if match is not None:
            paper.cited_by_count = match.cited_by_count
            paper.published = match.published or paper.published

        paper.paper_strength = compute_paper_strength(paper.cited_by_count, paper.published)

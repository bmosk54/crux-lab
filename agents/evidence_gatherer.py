"""Evidence gatherer: a Python-orchestrated pipeline over a single Claude model.

Pipeline stages:
  1. generate_queries  (LLM)    — frame 3 supporting + 3 refuting search queries
  2. collect_candidates (Python) — run the 6 searches, rank every hit
  3. select top 10 per side (Python) — deterministic, relevance-weighted
  4. fetch_abstracts   (Python) — pull abstracts for the ~20 finalists by ID
  5. extract_evidence  (LLM)    — read abstracts, extract & classify claims

Ranking (Python) blends relevance (highest weight), citation impact, and
title specificity, so the model never has to guess calibrated numbers.
"""

from __future__ import annotations

import json
import re

from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, TextBlock, query

from models.evidence import (
    RELEVANCE_FLOOR_ESTABLISHED,
    CandidatePaper,
    EvidenceSet,
    compute_composite,
    compute_paper_strength,
    compute_specificity,
    extract_keywords,
)
from tools.europepmc import fetch_abstracts, run_search

MODEL = "sonnet"
RELEVANCE_PAGE_SIZE = 20
ESTABLISHED_PAGE_SIZE = 10
ESTABLISHED_MIN_CITATIONS = 20
TOP_PER_SIDE = 10
ABSTRACT_CHAR_LIMIT = 1600

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


# ---------------------------------------------------------------------------
# Stage 5 — evidence extraction
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM = """
You are a rigorous scientific evidence analyst. You are given a hypothesis and a
set of candidate papers (already retrieved and ranked) with their abstracts.

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


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------


async def _complete(system: str, prompt: str) -> str:
    """Run a single-shot Claude completion (no tools) and return its text."""
    options = ClaudeAgentOptions(
        model=MODEL,
        system_prompt=system,
        max_turns=1,
        allowed_tools=[],
    )
    out = ""
    async for msg in query(prompt=prompt, options=options):
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    out += block.text
    return out


# ---------------------------------------------------------------------------
# Stage 1 — generate queries
# ---------------------------------------------------------------------------


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


async def generate_queries(hypothesis: str) -> tuple[list[str], list[str]]:
    text = await _complete(QUERY_GEN_SYSTEM, f"Hypothesis:\n\n{hypothesis}")
    match = re.search(r"<queries>\s*(.*?)\s*</queries>", text, re.DOTALL)
    if not match:
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1).strip())
            sup = [q for q in data.get("supporting_queries", []) if q.strip()]
            ref = [q for q in data.get("refuting_queries", []) if q.strip()]
            if sup and ref:
                return sup, ref
        except json.JSONDecodeError:
            pass
    return _fallback_queries(hypothesis)


# ---------------------------------------------------------------------------
# Stage 2-3 — search and rank
# ---------------------------------------------------------------------------


async def collect_candidates(
    hypothesis: str,
    supporting_queries: list[str],
    refuting_queries: list[str],
    top_per_side: int = TOP_PER_SIDE,
) -> tuple[list[CandidatePaper], list[CandidatePaper]]:
    """Run the discovery searches and rank all hits deterministically."""
    keywords = extract_keywords(hypothesis)
    candidates: dict[tuple[str, str], CandidatePaper] = {}

    def register(hit: dict, side: str, rank_score: float, established: bool) -> None:
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
        cand.best_rank = max(cand.best_rank, rank_score)
        cand.established = cand.established or established
        if side not in cand.pools:
            cand.pools.append(side)

    async def process(queries: list[str], side: str) -> None:
        for q in queries:
            # Pass A: relevance-sorted — captures best topical matches (incl. recent)
            rel = await run_search(q, page_size=RELEVANCE_PAGE_SIZE, sort="relevance")
            hits = rel.get("results", [])
            n = len(hits)
            for i, hit in enumerate(hits):
                register(hit, side, 1.0 - (i / n) if n else 0.0, established=False)

            # Pass B: citation-sorted with a floor — guarantees established,
            # high-impact papers enter the pool even if relevance buried them
            est = await run_search(
                q,
                page_size=ESTABLISHED_PAGE_SIZE,
                sort="citedByCount",
                min_citations=ESTABLISHED_MIN_CITATIONS,
            )
            for hit in est.get("results", []):
                register(hit, side, 0.0, established=True)

    await process(supporting_queries, "supporting")
    await process(refuting_queries, "refuting")

    for cand in candidates.values():
        relevance = cand.best_rank
        if cand.established:
            relevance = max(relevance, RELEVANCE_FLOOR_ESTABLISHED)
        relevance = min(1.0, relevance + 0.05 * (cand.found_count - 1))
        cand.relevance = round(relevance, 3)
        cand.specificity = compute_specificity(cand.title, keywords)
        cand.citation_score = compute_paper_strength(cand.cited_by_count, cand.published)
        cand.composite = compute_composite(cand.relevance, cand.citation_score, cand.specificity)

    supporting = sorted(
        (c for c in candidates.values() if "supporting" in c.pools),
        key=lambda c: c.composite,
        reverse=True,
    )[:top_per_side]
    refuting = sorted(
        (c for c in candidates.values() if "refuting" in c.pools),
        key=lambda c: c.composite,
        reverse=True,
    )[:top_per_side]
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


async def extract_evidence(hypothesis: str, papers: list[CandidatePaper]) -> EvidenceSet:
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
    text = await _complete(EXTRACTION_SYSTEM, prompt)
    evidence = _parse_evidence(text, hypothesis)
    _refresh_metadata(evidence, papers_with_abstracts)
    return evidence


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


async def gather_evidence(hypothesis: str, *, verbose: bool = True) -> EvidenceSet:
    def log(msg: str) -> None:
        if verbose:
            print(msg)

    log("  → generating search queries...")
    supporting_queries, refuting_queries = await generate_queries(hypothesis)

    n_queries = len(supporting_queries) + len(refuting_queries)
    log(f"  → running {n_queries * 2} discovery searches ({n_queries} queries × relevance+citation passes)...")
    supporting, refuting = await collect_candidates(
        hypothesis, supporting_queries, refuting_queries
    )

    log(f"  → ranked candidates: {len(supporting)} supporting, {len(refuting)} refuting")
    papers = await attach_abstracts(supporting, refuting)
    log(f"  → fetched abstracts for {sum(1 for p in papers if p.abstract.strip())} papers")

    log("  → reading abstracts and extracting evidence...")
    evidence = await extract_evidence(hypothesis, papers)
    return evidence


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_evidence(text: str, hypothesis: str) -> EvidenceSet:
    match = re.search(r"<evidence>\s*(.*?)\s*</evidence>", text, re.DOTALL)
    if not match:
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1).strip())
            return EvidenceSet(**data)
        except (json.JSONDecodeError, ValueError) as exc:
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

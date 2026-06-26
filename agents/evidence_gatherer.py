"""Evidence gatherer: a Python-orchestrated retrieval pipeline.

Stages (model allocation is a barbell — cheap for high-volume, premium for reasoning):
  0. decompose_claim    (premium, agents/decomposer) — runs upstream, passed in here
  1. generate_queries   (cheap)  — targeted supporting + refuting, driven by Stage 0
     + Python entity-verification queries (one per named_entity; drop the empties)
  2. collect_candidates (Python) — parallel fan-out search + minimum-relevance gate
  3. title_rank         (cheap)  — crude entity/topical filter, ~75 titles -> 30
  3b. abstract_pass     (cheap)  — score 30 abstracts on separate axes
  3c. select_working_set(Python) — reserved slots + claim-type-aware fill (no single score)
  3.5 attach_fulltext   (Python) — OA Results/Discussion for the pivotal 3-5, in parallel

The verdict (Stage 4) lives in agents/synthesizer.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import httpx

from agents.llm import complete, parse_tagged_json
from models.evidence import (
    CandidatePaper,
    Decomposition,
    Directness,
    EvidencePaper,
    EvidenceSet,
    EvidenceType,
    Stance,
    VenueQuality,
    citations_per_year,
    compute_paper_strength,
    compute_specificity,
    extract_keywords,
)
from tools.europepmc import fetch_abstracts, fetch_fulltext_sections, run_search

RELEVANCE_PAGE_SIZE = 20
ESTABLISHED_PAGE_SIZE = 10
ESTABLISHED_MIN_CITATIONS = 20
ENTITY_PAGE_SIZE = 8
MAX_ENTITY_QUERIES = 12
TITLE_SURVIVORS = 30
WORKING_SET_SIZE = 9
PIVOTAL_MAX = 4
ABSTRACT_CHAR_LIMIT = 1400


@dataclass
class GatherResult:
    """Everything produced by the gathering pipeline (for output + debugging)."""

    claim: str
    decomposition: Decomposition
    supporting_queries: list[str] = field(default_factory=list)
    refuting_queries: list[str] = field(default_factory=list)
    entity_queries: list[str] = field(default_factory=list)
    verified_entities: list[str] = field(default_factory=list)
    dropped_entities: list[str] = field(default_factory=list)
    candidates: list[CandidatePaper] = field(default_factory=list)
    evidence: EvidenceSet | None = None


# ---------------------------------------------------------------------------
# Stage 1 — query generation (driven by Stage 0 decomposition)
# ---------------------------------------------------------------------------

QUERY_GEN_SYSTEM = """
You generate Europe PMC search queries from a decomposed scientific claim. You receive
the claim plus its decomposition (sub-claims, load-bearing modifiers, claim_type, the
established knowledge it engages, and named entities).

Produce two sets.

supporting_queries (3): likely to surface evidence the claim is TRUE — positive
findings, supportive mechanisms, successful interventions on the claim's actual
variables. Make the three differ in angle (mechanism / population / study type), and
let at least one target recent work.

refuting_queries (3): each has a SPECIFIC job — do not write three generic "evidence
against" queries:
  1. ESTABLISHED ALTERNATIVE / UPSTREAM MODEL — retrieve the consensus position named in
     established_knowledge (the model this claim displaces, sits beneath, or duplicates).
  2. NULL / FAILED-REPLICATION on the claim's specific intervention or entity — explicitly
     target null results, "no effect", "failed to replicate", "did not reproduce",
     negative trials for THIS claim's mechanism/drug/program.
  3. GENERAL COUNTER — limitations, controversy, contradictory findings.

Guidelines:
- Use precise terminology and the claim's key variables/entities.
- You may use Europe PMC field syntax ("exact phrase", MESH:Term).

Output EXACTLY this and nothing else:

<queries>
{"supporting_queries": ["q1","q2","q3"], "refuting_queries": ["q1","q2","q3"]}
</queries>
""".strip()


def _decomp_brief(d: Decomposition) -> str:
    return (
        f"Claim: {d.claim}\n"
        f"Sub-claims: {'; '.join(d.sub_claims) or '(none)'}\n"
        f"Load-bearing modifiers: {', '.join(d.load_bearing_modifiers) or '(none)'}\n"
        f"Claim type: {d.claim_type}\n"
        f"Established knowledge engaged: {d.established_knowledge or '(none)'}\n"
        f"Named entities (candidate): {', '.join(d.named_entities) or '(none)'}"
    )


def _fallback_queries(d: Decomposition) -> tuple[list[str], list[str]]:
    c = d.claim
    return (
        [c, f"{c} mechanism evidence", f"{c} recent findings"],
        [
            d.established_knowledge or f"{c} established mechanism",
            f"{c} null result OR failed replication OR no effect",
            f"{c} limitations OR controversy OR contradictory",
        ],
    )


def _build_entity_queries(d: Decomposition) -> list[tuple[str, str]]:
    """One verification query per named entity: (entity, query_string)."""
    topic = sorted(extract_keywords(d.claim), key=len, reverse=True)[:2]
    topic_str = " ".join(topic)
    out: list[tuple[str, str]] = []
    for entity in d.named_entities[:MAX_ENTITY_QUERIES]:
        entity = entity.strip()
        if not entity:
            continue
        q = f'"{entity}" {topic_str}'.strip()
        out.append((entity, q))
    return out


async def generate_queries(
    decomposition: Decomposition, *, model: str = "haiku"
) -> tuple[list[str], list[str]]:
    text = await complete(QUERY_GEN_SYSTEM, _decomp_brief(decomposition), model=model)
    data = parse_tagged_json(text, "queries")
    if isinstance(data, dict):
        sup = [q for q in data.get("supporting_queries", []) if isinstance(q, str) and q.strip()]
        ref = [q for q in data.get("refuting_queries", []) if isinstance(q, str) and q.strip()]
        if sup and ref:
            return sup, ref
    return _fallback_queries(decomposition)


# ---------------------------------------------------------------------------
# Stage 2 — discovery (parallel fan-out) + minimum-relevance gate
# ---------------------------------------------------------------------------


async def collect_candidates(
    decomposition: Decomposition,
    supporting_queries: list[str],
    refuting_queries: list[str],
    entity_queries: list[tuple[str, str]],
) -> tuple[list[CandidatePaper], list[str], list[str]]:
    """Run all searches concurrently, dedupe, gate, and verify entities.

    Returns (gated_candidates, verified_entities, dropped_entities).
    """
    keywords = extract_keywords(decomposition.claim + " " + " ".join(decomposition.sub_claims))
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
                is_open_access=bool(hit.get("is_open_access")),
            )
            candidates[key] = cand
        cand.found_count += 1
        cand.established = cand.established or established
        if side not in cand.pools:
            cand.pools.append(side)

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Build every search coroutine up front, then fan out in parallel.
        jobs: list[tuple[str, bool, object]] = []  # (side, established, coro)
        for q in supporting_queries:
            jobs.append(("supporting", False, run_search(q, page_size=RELEVANCE_PAGE_SIZE, sort="relevance", client=client)))
            jobs.append(("supporting", True, run_search(q, page_size=ESTABLISHED_PAGE_SIZE, sort="citedByCount", min_citations=ESTABLISHED_MIN_CITATIONS, client=client)))
        for q in refuting_queries:
            jobs.append(("refuting", False, run_search(q, page_size=RELEVANCE_PAGE_SIZE, sort="relevance", client=client)))
            jobs.append(("refuting", True, run_search(q, page_size=ESTABLISHED_PAGE_SIZE, sort="citedByCount", min_citations=ESTABLISHED_MIN_CITATIONS, client=client)))

        entity_jobs = [
            (entity, run_search(q, page_size=ENTITY_PAGE_SIZE, sort="relevance", client=client))
            for entity, q in entity_queries
        ]

        all_results = await asyncio.gather(
            *(coro for _, _, coro in jobs),
            *(coro for _, coro in entity_jobs),
            return_exceptions=True,
        )

    n_topic = len(jobs)
    topic_results = all_results[:n_topic]
    entity_results = all_results[n_topic:]

    for (side, established, _), res in zip(jobs, topic_results):
        if isinstance(res, dict):
            for hit in res.get("results", []):
                register(hit, side, established)

    verified: list[str] = []
    dropped: list[str] = []
    for (entity, _), res in zip(entity_jobs, entity_results):
        hits = res.get("results", []) if isinstance(res, dict) else []
        if hits:
            verified.append(entity)
            for hit in hits:
                register(hit, "entity", established=False)
        else:
            dropped.append(entity)  # zero hits -> hallucination-safe drop

    # Minimum-relevance gate: keep if the title shares a hypothesis keyword, OR the
    # paper is an established high-impact hit, OR it was surfaced by a verified entity.
    gated: list[CandidatePaper] = []
    for cand in candidates.values():
        cand.specificity = compute_specificity(cand.title, keywords)
        if cand.specificity > 0 or cand.established or "entity" in cand.pools:
            gated.append(cand)
    return gated, verified, dropped


# ---------------------------------------------------------------------------
# Stage 3 — title pass (crude entity/topical filter, ~75 -> 30)
# ---------------------------------------------------------------------------

TITLE_RANK_SYSTEM = """
You are doing a CRUDE title-level triage. You see only titles (plus year, citations,
and which search surfaced each). Titles cannot tell you whether a paper supports or
refutes — do NOT try. Judge only two things:
- ENTITY MATCH: does the title mention the claim's specific gene/drug/program/method,
  or its actual variables (not just the broad field)?
- TOPICAL RELEVANCE: is this plausibly about the claim's subject at all?

Keep the up-to-30 titles most worth reading in full. Keep entity-specific and
established/consensus titles even if phrased differently. Drop titles about a clearly
different disease/organism/method with no link, and pure assay/tooling papers. It is
fine to keep fewer than 30.

Output EXACTLY this and nothing else:

<selection>
{"keep": [paper numbers]}
</selection>
""".strip()


def _format_title_line(index: int, cand: CandidatePaper) -> str:
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


async def title_rank(
    decomposition: Decomposition,
    candidates: list[CandidatePaper],
    limit: int = TITLE_SURVIVORS,
    *,
    model: str = "haiku",
) -> list[CandidatePaper]:
    if not candidates:
        return []
    if len(candidates) <= limit:
        return candidates

    ordered = sorted(candidates, key=lambda c: c.cited_by_count, reverse=True)
    listing = "\n".join(_format_title_line(i, c) for i, c in enumerate(ordered, 1))
    prompt = (
        f"Claim: {decomposition.claim}\n"
        f"Named entities: {', '.join(decomposition.named_entities) or '(none)'}\n\n"
        f"Candidate titles ({len(ordered)}):\n{listing}"
    )
    text = await complete(TITLE_RANK_SYSTEM, prompt, model=model)
    data = parse_tagged_json(text, "selection") or {}
    keep = _coerce_indices(data.get("keep"), len(ordered))[:limit]
    if not keep:
        # Fallback: most-cited + any entity/established papers.
        keep = [i for i, c in enumerate(ordered, 1) if c.established or "entity" in c.pools]
        keep = (keep + [i for i in range(1, len(ordered) + 1) if i not in keep])[:limit]
    return [ordered[i - 1] for i in keep]


# ---------------------------------------------------------------------------
# Stage 4 — fetch abstracts for the 30 survivors
# ---------------------------------------------------------------------------


async def attach_abstracts(candidates: list[CandidatePaper]) -> list[CandidatePaper]:
    union = {(c.source, c.ext_id): c for c in candidates}
    fetched = await fetch_abstracts(list(union.keys()))
    for (_source, ext_id), cand in union.items():
        record = fetched.get(ext_id)
        if record and record.get("abstract"):
            cand.abstract = record["abstract"]
    return list(union.values())


# ---------------------------------------------------------------------------
# Stage 3b — abstract pass (cheap; score each paper on SEPARATE axes)
# ---------------------------------------------------------------------------

ABSTRACT_PASS_SYSTEM = """
You read abstracts and score each paper on several SEPARATE axes. Do not blend them
into one number. You receive the claim, its decomposition, and numbered papers with
abstracts.

For EACH paper output:
- stance: does the abstract's finding "support" / "refute" / "mixed" the claim — or
  "irrelevant" if on reflection it does not bear on the claim. Mark irrelevant freely;
  irrelevant papers are dropped.
- directness: "direct" (tests the claim's ACTUAL variables), "indirect" (tests a proxy
  / surrogate / one step away), or "tangential" (different context, multi-step link).
- evidence_tier: one of rct | meta_analysis | cohort | animal_model | in_vitro |
  mechanistic | case_report | review.
- entity_specific: true if it names the claim's actual gene/drug/program/method (not a
  generic class).
- venue_quality: "high" (strong primary venue), "ok", or "low" (predatory, or a
  narrative review merely advocating a contested hypothesis — not a peer result).
- is_consensus_paper: true if this paper states/represents the established knowledge /
  reigning paradigm the claim engages.
- key_findings: 1-3 short, concrete findings (with direction/effect), not vague themes.

Output EXACTLY this and nothing else:

<analysis>
{"papers": [
  {"n": 1, "stance": "refute", "directness": "direct", "evidence_tier": "rct",
   "entity_specific": true, "venue_quality": "high", "is_consensus_paper": false,
   "key_findings": ["..."]}
]}
</analysis>
""".strip()


def _format_abstract_block(index: int, cand: CandidatePaper) -> str:
    abstract = (cand.abstract or "").strip()
    if len(abstract) > ABSTRACT_CHAR_LIMIT:
        abstract = abstract[:ABSTRACT_CHAR_LIMIT] + "…"
    year = (cand.published or "?")[:4]
    return (
        f"[Paper {index}] ({cand.cited_by_count} cit · {year})\n"
        f"title: {cand.title}\n"
        f"abstract: {abstract or '(no abstract available)'}\n"
    )


_VALID_STANCE = {"support", "refute", "mixed"}
_VALID_DIRECTNESS = {"direct", "indirect", "tangential"}
_VALID_TIER = {"rct", "meta_analysis", "cohort", "animal_model", "in_vitro", "mechanistic", "case_report", "review"}
_VALID_VENUE = {"high", "ok", "low"}


def _build_evidence_paper(cand: CandidatePaper, axes: dict) -> EvidencePaper:
    stance: Stance = axes.get("stance") if axes.get("stance") in _VALID_STANCE else "mixed"
    directness: Directness = axes.get("directness") if axes.get("directness") in _VALID_DIRECTNESS else "indirect"
    tier: EvidenceType = axes.get("evidence_tier") if axes.get("evidence_tier") in _VALID_TIER else "review"
    venue: VenueQuality = axes.get("venue_quality") if axes.get("venue_quality") in _VALID_VENUE else "ok"
    findings = [f for f in axes.get("key_findings", []) if isinstance(f, str) and f.strip()][:3]
    snippet = findings[0] if findings else (cand.abstract or "")[:240]
    return EvidencePaper(
        title=cand.title,
        authors=cand.authors,
        journal=cand.journal,
        published=cand.published,
        pmid=cand.pmid,
        pmcid=cand.pmcid,
        doi=cand.doi,
        cited_by_count=cand.cited_by_count,
        stance=stance,
        directness=directness,
        evidence_tier=tier,
        entity_specific=bool(axes.get("entity_specific")),
        venue_quality=venue,
        is_consensus_paper=bool(axes.get("is_consensus_paper")),
        key_findings=findings,
        abstract_snippet=snippet,
        paper_strength=compute_paper_strength(cand.cited_by_count, cand.published),
    )


ABSTRACT_PASS_CHUNK = 10


async def _abstract_pass_chunk(
    decomposition: Decomposition, chunk: list[CandidatePaper], model: str
) -> list[EvidencePaper]:
    blocks = "\n".join(_format_abstract_block(i, c) for i, c in enumerate(chunk, 1))
    prompt = (
        f"{_decomp_brief(decomposition)}\n\n"
        f"Papers ({len(chunk)}):\n\n{blocks}"
    )
    text = await complete(ABSTRACT_PASS_SYSTEM, prompt, model=model)
    data = parse_tagged_json(text, "analysis") or {}
    rows = data.get("papers", []) if isinstance(data, dict) else []

    by_n: dict[int, dict] = {}
    for row in rows:
        if isinstance(row, dict):
            try:
                by_n[int(row.get("n"))] = row
            except (TypeError, ValueError):
                continue

    papers: list[EvidencePaper] = []
    for i, cand in enumerate(chunk, 1):
        axes = by_n.get(i)
        if axes is None or axes.get("stance") == "irrelevant":
            continue
        papers.append(_build_evidence_paper(cand, axes))
    return papers


async def abstract_pass(
    decomposition: Decomposition,
    candidates: list[CandidatePaper],
    *,
    model: str = "haiku",
) -> list[EvidencePaper]:
    """Score abstracts on separate axes. Chunked into small PARALLEL calls so the
    cheap model's structured output never overruns a single turn."""
    with_abstracts = [c for c in candidates if c.abstract.strip()]
    if not with_abstracts:
        return []

    chunks = [
        with_abstracts[i : i + ABSTRACT_PASS_CHUNK]
        for i in range(0, len(with_abstracts), ABSTRACT_PASS_CHUNK)
    ]
    results = await asyncio.gather(
        *(_abstract_pass_chunk(decomposition, ch, model) for ch in chunks)
    )
    papers: list[EvidencePaper] = []
    for r in results:
        papers.extend(r)
    return papers


# ---------------------------------------------------------------------------
# Stage 3c — working-set selection (Python; reserved slots, NOT pure top-N)
# ---------------------------------------------------------------------------

_DIRECTNESS_RANK = {"direct": 3, "indirect": 2, "tangential": 1}
_VENUE_RANK = {"high": 2, "ok": 1, "low": 0}
_TIER_BASE = {
    "meta_analysis": 6, "rct": 5, "cohort": 4, "animal_model": 3,
    "in_vitro": 2, "case_report": 1, "mechanistic": 1, "review": 0,
}


def _tier_score(tier: str, claim_type: str) -> int:
    """Evidence-tier value, re-weighted by claim type (Change 3)."""
    base = _TIER_BASE.get(tier, 0)
    if claim_type in ("mechanistic", "functional"):
        if tier in ("in_vitro", "animal_model"):
            base += 3
        if tier == "mechanistic":
            base += 1
    elif claim_type in ("therapeutic", "causal"):
        if tier in ("rct", "meta_analysis"):
            base += 2
        if tier == "cohort":
            base += 1
    elif claim_type == "association":
        if tier in ("cohort", "meta_analysis"):
            base += 2
    return base


def _ordering_key(p: EvidencePaper, claim_type: str) -> tuple:
    """Lexicographic axis ordering — directness first, impact only as last tiebreak."""
    return (
        _DIRECTNESS_RANK.get(p.directness, 0),
        _tier_score(p.evidence_tier, claim_type),
        1 if p.entity_specific else 0,
        _VENUE_RANK.get(p.venue_quality, 1),
        citations_per_year(p.cited_by_count, p.published),
    )


def select_working_set(
    papers: list[EvidencePaper],
    decomposition: Decomposition,
    size: int = WORKING_SET_SIZE,
) -> EvidenceSet:
    claim_type = decomposition.claim_type
    es = EvidenceSet(hypothesis=decomposition.claim)
    if not papers:
        return es

    chosen: list[EvidencePaper] = []
    chosen_ids: set[int] = set()

    def take(p: EvidencePaper | None, reason: str) -> None:
        if p is None or id(p) in chosen_ids:
            return
        if p.reserved_as is None:
            p.reserved_as = reason
        chosen.append(p)
        chosen_ids.add(id(p))

    # --- Reserved slots: guarantee these survive even if "mid" by composite ---
    refuters = [p for p in papers if p.stance == "refute"]
    if refuters:
        take(max(refuters, key=lambda p: _ordering_key(p, claim_type)), "highest-directness refuting")

    consensus = [p for p in papers if p.is_consensus_paper]
    if consensus:
        take(max(consensus, key=lambda p: p.cited_by_count), "established_knowledge / consensus")

    specific = [p for p in papers if p.entity_specific]
    if specific:
        take(max(specific, key=lambda p: _ordering_key(p, claim_type)), "most entity-specific")

    # --- Fill remaining by claim-type-aware lexicographic ordering ---
    rest = sorted(
        (p for p in papers if id(p) not in chosen_ids),
        key=lambda p: _ordering_key(p, claim_type),
        reverse=True,
    )
    for p in rest:
        if len(chosen) >= size:
            break
        take(p, "")

    # --- Light stance balance: ensure both sides are represented if available ---
    _balance_stances(chosen, papers, claim_type, size)

    es.papers = chosen[:size]
    return es


def _balance_stances(
    chosen: list[EvidencePaper],
    papers: list[EvidencePaper],
    claim_type: str,
    size: int,
    min_each: int = 2,
) -> None:
    chosen_ids = {id(p) for p in chosen}
    for stance in ("support", "refute"):
        have = [p for p in chosen if p.stance == stance]
        if len(have) >= min_each:
            continue
        pool = sorted(
            (p for p in papers if p.stance == stance and id(p) not in chosen_ids),
            key=lambda p: _ordering_key(p, claim_type),
            reverse=True,
        )
        for cand in pool:
            if len([p for p in chosen if p.stance == stance]) >= min_each:
                break
            # Drop the weakest non-reserved paper of the over-represented side.
            droppable = [
                p for p in chosen
                if p.reserved_as is None and p.stance != stance
            ]
            if not droppable or len(chosen) < size:
                chosen.append(cand)
                chosen_ids.add(id(cand))
            else:
                weakest = min(droppable, key=lambda p: _ordering_key(p, claim_type))
                chosen.remove(weakest)
                chosen_ids.discard(id(weakest))
                chosen.append(cand)
                chosen_ids.add(id(cand))


# ---------------------------------------------------------------------------
# Stage 3.5 — selective full text for pivotal papers (parallel)
# ---------------------------------------------------------------------------


def _pivotal_papers(evidence: EvidenceSet, claim_type: str) -> list[EvidencePaper]:
    """The 3-5 papers the verdict hinges on: reserved slots + highest directness."""
    reserved = [p for p in evidence.papers if p.reserved_as]
    rest = sorted(
        (p for p in evidence.papers if not p.reserved_as),
        key=lambda p: _ordering_key(p, claim_type),
        reverse=True,
    )
    ordered = reserved + rest
    return ordered[:PIVOTAL_MAX]


async def attach_fulltext(evidence: EvidenceSet, decomposition: Decomposition) -> int:
    """Fetch OA Results/Discussion for pivotal papers in parallel. Returns count attached."""
    pivotal = _pivotal_papers(evidence, decomposition.claim_type)
    by_pmcid = {p.pmcid: p for p in pivotal if p.pmcid}
    if not by_pmcid:
        return 0
    sections = await fetch_fulltext_sections(list(by_pmcid.keys()))
    n = 0
    for pmcid, text in sections.items():
        paper = by_pmcid.get(pmcid)
        if paper is not None:
            paper.fulltext_sections = text
            paper.abstract_only = False
            n += 1
    return n


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


async def gather_evidence(
    decomposition: Decomposition,
    *,
    verbose: bool = True,
    model_query_gen: str = "haiku",
    model_title_rank: str = "haiku",
    model_abstract_pass: str = "haiku",
) -> GatherResult:
    def log(msg: str) -> None:
        if verbose:
            print(msg)

    log(f"  → generating targeted queries (model: {model_query_gen})...")
    supporting_queries, refuting_queries = await generate_queries(
        decomposition, model=model_query_gen
    )
    entity_queries = _build_entity_queries(decomposition)

    n_topic = (len(supporting_queries) + len(refuting_queries)) * 2
    log(f"  → fanning out {n_topic + len(entity_queries)} searches in parallel...")
    candidates, verified, dropped = await collect_candidates(
        decomposition, supporting_queries, refuting_queries, entity_queries
    )
    if dropped:
        log(f"  → dropped unverified entities (0 hits): {', '.join(dropped)}")
    log(f"  → {len(candidates)} candidates passed the relevance gate")

    log(f"  → title pass -> {TITLE_SURVIVORS} (model: {model_title_rank})...")
    survivors = await title_rank(decomposition, candidates, model=model_title_rank)
    log(f"  → {len(survivors)} titles survived")

    survivors = await attach_abstracts(survivors)
    n_abs = sum(1 for c in survivors if c.abstract.strip())
    log(f"  → fetched abstracts for {n_abs} papers")

    log(f"  → abstract pass scoring axes (model: {model_abstract_pass})...")
    analyzed = await abstract_pass(decomposition, survivors, model=model_abstract_pass)
    log(f"  → {len(analyzed)} relevant papers scored")

    evidence = select_working_set(analyzed, decomposition)
    log(f"  → selected working set of {len(evidence.papers)} papers")

    n_ft = await attach_fulltext(evidence, decomposition)
    log(f"  → pulled full-text Results/Discussion for {n_ft} pivotal papers")

    return GatherResult(
        claim=decomposition.claim,
        decomposition=decomposition,
        supporting_queries=supporting_queries,
        refuting_queries=refuting_queries,
        entity_queries=[q for _, q in entity_queries],
        verified_entities=verified,
        dropped_entities=dropped,
        candidates=survivors,
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# Change 6 — optional single coverage gate (≤1 extra search batch + resynth)
# ---------------------------------------------------------------------------


def _paper_key(p) -> tuple:
    return (
        (p.doi or "").lower() or None,
        p.pmid or None,
        p.title.lower(),
    )


def _coverage_gaps(result: GatherResult) -> list[str]:
    """Verified entities the working set never actually mentions."""
    evidence = result.evidence
    if evidence is None:
        return []
    blob = " ".join(
        (p.title + " " + " ".join(p.key_findings)).lower() for p in evidence.papers
    )
    return [e for e in result.verified_entities if e.strip() and e.lower() not in blob]


async def run_coverage_gate(
    decomposition: Decomposition,
    result: GatherResult,
    verdict,
    *,
    verbose: bool = True,
    model_abstract_pass: str = "haiku",
    model_synthesis: str = "sonnet",
    synthesize_fn=None,
):
    """Fire AT MOST one targeted search + re-synthesis if a verified entity is missing.

    Safety net only — if Stage 0 worked this rarely fires. Mutates result.evidence
    in place and returns the (possibly updated) verdict.
    """
    def log(msg: str) -> None:
        if verbose:
            print(msg)

    evidence = result.evidence
    if evidence is None or synthesize_fn is None:
        return verdict

    gaps = _coverage_gaps(result)
    if not gaps:
        log("  → coverage gate: no missing entities; skipping")
        return verdict

    log(f"  → coverage gate FIRED for missing entities: {', '.join(gaps)}")
    topic = " ".join(sorted(extract_keywords(decomposition.claim), key=len, reverse=True)[:2])

    new_candidates: dict[tuple[str, str], CandidatePaper] = {}
    async with httpx.AsyncClient(timeout=30.0) as client:
        jobs = [run_search(f'"{e}" {topic}', page_size=ENTITY_PAGE_SIZE, sort="relevance", client=client) for e in gaps]
        results = await asyncio.gather(*jobs, return_exceptions=True)
    for res in results:
        if not isinstance(res, dict):
            continue
        for hit in res.get("results", []):
            source, ext_id = hit.get("source"), hit.get("id")
            if not source or not ext_id:
                continue
            new_candidates[(source, ext_id)] = CandidatePaper(
                source=source, ext_id=ext_id, title=hit.get("title") or "",
                authors=hit.get("authors") or "", journal=hit.get("journal"),
                published=hit.get("published"), pmid=hit.get("pmid"),
                pmcid=hit.get("pmcid"), doi=hit.get("doi"),
                cited_by_count=int(hit.get("cited_by_count") or 0),
                is_open_access=bool(hit.get("is_open_access")), pools=["entity"],
            )

    if not new_candidates:
        log("  → coverage gate: targeted search returned nothing new")
        return verdict

    enriched = await attach_abstracts(list(new_candidates.values()))
    analyzed = await abstract_pass(decomposition, enriched, model=model_abstract_pass)

    existing_keys = {_paper_key(p) for p in evidence.papers}
    additions = [p for p in analyzed if _paper_key(p) not in existing_keys]
    additions.sort(key=lambda p: _ordering_key(p, decomposition.claim_type), reverse=True)
    additions = additions[:4]
    if not additions:
        log("  → coverage gate: no new relevant papers after scoring")
        return verdict

    evidence.papers.extend(additions)
    await attach_fulltext(evidence, decomposition)
    log(f"  → coverage gate added {len(additions)} paper(s); re-synthesizing once")

    return await synthesize_fn(
        decomposition.claim, decomposition, evidence,
        verbose=verbose, model=model_synthesis,
    )

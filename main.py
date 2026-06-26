"""crux-lab — decompose a claim, gather evidence, and render a calibrated verdict."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import anyio
from dotenv import load_dotenv

from agents.decomposer import decompose_claim
from agents.evidence_gatherer import GatherResult, gather_evidence, run_coverage_gate
from agents.synthesizer import synthesize
from config import DEFAULT_CONFIG, EXAMPLE_CLAIMS, PipelineConfig
from models.evidence import (
    CandidatePaper,
    Decomposition,
    EvidencePaper,
    EvidenceSet,
    Verdict,
)

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

WIDTH = 72
_STARS = {(0.0, 0.3): "★☆☆☆☆", (0.3, 0.5): "★★☆☆☆", (0.5, 0.7): "★★★☆☆", (0.7, 0.9): "★★★★☆", (0.9, 1.01): "★★★★★"}


def _stars(score: float) -> str:
    for (lo, hi), label in _STARS.items():
        if lo <= score < hi:
            return label
    return "★☆☆☆☆"


def _header(title: str) -> None:
    print("\n" + "═" * WIDTH)
    print(f"  {title}")
    print("═" * WIDTH)


# ---------------------------------------------------------------------------
# Step displays (each gated by a config flag)
# ---------------------------------------------------------------------------


def display_decomposition(d: Decomposition) -> None:
    _header("STEP 0 · DECOMPOSITION")
    print(f"\n  Claim type: {d.claim_type}")
    print("\n  Sub-claims:")
    for i, s in enumerate(d.sub_claims, 1):
        print(f"    {i}. {s}")
    print(f"\n  Load-bearing modifiers: {', '.join(d.load_bearing_modifiers) or '(none)'}")
    print(f"\n  Established knowledge engaged:\n    {d.established_knowledge or '(none)'}")
    print(f"\n  Named entities (candidate): {', '.join(d.named_entities) or '(none)'}")


def display_queries(result: GatherResult) -> None:
    _header("STEP 1 · SEARCH QUERIES")
    print("\n  Supporting:")
    for q in result.supporting_queries:
        print(f"    • {q}")
    print("\n  Refuting (targeted):")
    for q in result.refuting_queries:
        print(f"    • {q}")
    if result.entity_queries:
        print("\n  Entity-verification:")
        for q in result.entity_queries:
            print(f"    • {q}")
    if result.verified_entities:
        print(f"\n  Verified entities: {', '.join(result.verified_entities)}")
    if result.dropped_entities:
        print(f"  Dropped (0 hits): {', '.join(result.dropped_entities)}")


def display_candidates(result: GatherResult) -> None:
    _header("STEP 2 · TITLE-PASS SURVIVORS")
    print(f"\n  {len(result.candidates)} titles survived to the abstract pass:")
    for cand in result.candidates:
        year = (cand.published or "?")[:4]
        print(f"    • [{cand.cited_by_count} cit · {year}] {cand.title}")


def _print_paper(paper: EvidencePaper, index: int) -> None:
    tags = f"{paper.stance.upper()} · {paper.evidence_tier} · {paper.directness}"
    if paper.entity_specific:
        tags += " · entity-specific"
    if paper.is_consensus_paper:
        tags += " · CONSENSUS"
    depth = "full-text" if not paper.abstract_only else "abstract-only"
    print(f"\n  {index}. [{_stars(paper.paper_strength)} {paper.paper_strength:.2f}] {paper.title}")
    print(f"     {tags} · {depth} · venue:{paper.venue_quality}")
    if paper.reserved_as:
        print(f"     ⮑ reserved slot: {paper.reserved_as}")
    print(f"     {paper.authors}")
    print(f"     {paper.journal or 'Unknown journal'} · {(paper.published or '')[:4] or 'n/a'} · {paper.cited_by_count} citations")
    if paper.doi:
        print(f"     https://doi.org/{paper.doi}")
    elif paper.pmid:
        print(f"     https://pubmed.ncbi.nlm.nih.gov/{paper.pmid}")
    for f in paper.key_findings:
        print(f"       • {f}")


def display_evidence(evidence: EvidenceSet) -> None:
    _header("STEP 3 · EVIDENCE (working set)")
    sup = evidence.by_stance("support")
    ref = evidence.by_stance("refute")
    mix = evidence.by_stance("mixed")
    print(f"\n📗 SUPPORTING ({len(sup)})   📕 REFUTING ({len(ref)})   📘 MIXED ({len(mix)})")
    print("─" * WIDTH)
    if not evidence.papers:
        print("  No usable evidence found.")
        return
    idx = 1
    for paper in [*sup, *ref, *mix]:
        _print_paper(paper, idx)
        idx += 1


def display_verdict(verdict: Verdict) -> None:
    _header("STEP 4 · VERDICT")
    print(
        f"\n  Claim as written: [{_stars(verdict.support_score)} {verdict.support_score:.2f}]"
        f"  ·  {verdict.overall_label.upper()}  ·  confidence {verdict.overall_confidence}/100"
    )
    if verdict.absence_coverage_note:
        print(f"\n  ⚠ {verdict.absence_coverage_note}")

    if verdict.sub_claims:
        print("\n  Per sub-claim:")
        for sc in verdict.sub_claims:
            print(f"    • [{sc.label.upper()} · {sc.confidence}/100] {sc.sub_claim}")
            if sc.basis:
                print(f"        {sc.basis}")

    if verdict.consensus_positioning:
        print("\n  Position vs. consensus:")
        print(f"    {verdict.consensus_positioning}")

    if verdict.counter_evidence_assessment:
        print("\n  Counter-evidence (applicability-adjusted):")
        print(f"    {verdict.counter_evidence_assessment}")

    if verdict.steelman.statement:
        gap = verdict.steelman.confidence - verdict.overall_confidence
        gap_note = f"  (+{gap} vs literal)" if gap > 0 else ""
        print(f"\n  Steelman [{verdict.steelman.confidence}/100{gap_note}]:")
        print(f"    {verdict.steelman.statement}")

    if verdict.untested_components:
        print("\n  Untested / not isolated by the evidence:")
        for u in verdict.untested_components:
            print(f"    • {u}")

    if verdict.decisive_experiment:
        print("\n  Decisive experiment:")
        print(f"    {verdict.decisive_experiment}")

    if verdict.reasoning:
        print("\n  Overall:")
        print(f"    {verdict.reasoning}")
    print("\n" + "═" * WIDTH + "\n")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


async def run(claim: str, config: PipelineConfig = DEFAULT_CONFIG) -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to .env in the project root.")

    print("\n🔬 Working through the claim (this may take a couple of minutes):\n")
    timings: dict[str, float] = {}

    t = time.time()
    decomposition = await decompose_claim(
        claim, model=config.model_decomposition, verbose=config.verbose
    )
    timings["decompose"] = time.time() - t
    if config.show_decomposition:
        display_decomposition(decomposition)

    t = time.time()
    result = await gather_evidence(
        decomposition,
        verbose=config.verbose,
        model_query_gen=config.model_query_gen,
        model_title_rank=config.model_title_rank,
        model_abstract_pass=config.model_abstract_pass,
    )
    timings["gather"] = time.time() - t

    if config.show_queries:
        display_queries(result)
    if config.show_candidates:
        display_candidates(result)
    if config.show_evidence and result.evidence is not None:
        display_evidence(result.evidence)

    if not (config.run_synthesis and result.evidence is not None):
        return

    t = time.time()
    verdict = await synthesize(
        claim, decomposition, result.evidence,
        verbose=config.verbose, model=config.model_synthesis,
    )
    timings["synthesize"] = time.time() - t

    # Change 6 — optional single coverage gate (≤1 extra search + resynth).
    if config.coverage_gate:
        verdict = await run_coverage_gate(
            decomposition, result, verdict,
            verbose=config.verbose,
            model_abstract_pass=config.model_abstract_pass,
            model_synthesis=config.model_synthesis,
            synthesize_fn=synthesize,
        )
        if config.show_evidence:
            display_evidence(result.evidence)

    if config.show_verdict:
        display_verdict(verdict)

    if config.verbose:
        breakdown = "  ".join(f"{k}={v:.0f}s" for k, v in timings.items())
        print(f"  ⏱  {breakdown}  total={sum(timings.values()):.0f}s")


def main() -> None:
    print("\n╔══════════════════════════════════╗")
    print("║   crux-lab · claim adjudicator   ║")
    print("╚══════════════════════════════════╝")
    print("\nEnter a scientific claim and I'll decompose it, gather evidence,")
    print("and render a calibrated verdict.\n")
    print("Example claims you can try:")
    for claim in EXAMPLE_CLAIMS[:5]:
        print(f"  • {claim}")
    print()

    claim = input("Claim: ").strip()
    if not claim:
        print("No claim entered. Exiting.")
        sys.exit(0)

    anyio.run(run, claim)


if __name__ == "__main__":
    main()

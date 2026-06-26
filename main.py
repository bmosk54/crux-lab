"""crux-lab — gather evidence, synthesize it, and render a verdict on a claim."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import anyio
from dotenv import load_dotenv

from agents.evidence_gatherer import GatherResult, gather_evidence
from agents.synthesizer import synthesize
from agents.verdict import generate_verdict
from config import DEFAULT_CONFIG, EXAMPLE_CLAIMS, PipelineConfig
from models.evidence import (
    CandidatePaper,
    EvidencePaper,
    EvidenceSet,
    Synthesis,
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


def display_queries(result: GatherResult) -> None:
    _header("STEP 1 · SEARCH QUERIES")
    print("\n  Supporting:")
    for q in result.supporting_queries:
        print(f"    • {q}")
    print("\n  Refuting:")
    for q in result.refuting_queries:
        print(f"    • {q}")


def _candidate_line(cand: CandidatePaper) -> str:
    year = (cand.published or "?")[:4]
    return f"    • [{cand.cited_by_count} cit · {year}] {cand.title}"


def display_candidates(result: GatherResult) -> None:
    _header("STEP 2 · SCREENED CANDIDATES")
    print(f"\n  Supporting candidates ({len(result.supporting_candidates)}):")
    for cand in result.supporting_candidates:
        print(_candidate_line(cand))
    print(f"\n  Refuting candidates ({len(result.refuting_candidates)}):")
    for cand in result.refuting_candidates:
        print(_candidate_line(cand))


def _print_paper(paper: EvidencePaper, index: int) -> None:
    print(f"\n  {index}. [{_stars(paper.paper_strength)}  {paper.paper_strength:.2f}] {paper.title}")
    print(f"     {paper.authors}")
    print(f"     {paper.journal or 'Unknown journal'} · {(paper.published or '')[:4] or 'n/a'} · {paper.cited_by_count} citations")
    if paper.doi:
        print(f"     https://doi.org/{paper.doi}")
    elif paper.pmid:
        print(f"     https://pubmed.ncbi.nlm.nih.gov/{paper.pmid}")
    if paper.abstract_snippet:
        print(f'\n     "{paper.abstract_snippet}"')
    if paper.supporting_claims:
        print("\n     Supporting claims:")
        for c in paper.supporting_claims:
            print(f"       + [{c.evidence_type} · {c.directness}] {c.claim}")
    if paper.refuting_claims:
        print("\n     Refuting claims:")
        for c in paper.refuting_claims:
            print(f"       - [{c.evidence_type} · {c.directness}] {c.claim}")


def display_evidence(evidence: EvidenceSet) -> None:
    _header("STEP 3 · EVIDENCE")
    print(f"\n📗 SUPPORTING EVIDENCE ({len(evidence.supporting_papers)} papers)")
    print("─" * WIDTH)
    if evidence.supporting_papers:
        for i, paper in enumerate(evidence.supporting_papers, 1):
            _print_paper(paper, i)
    else:
        print("  No supporting papers found.")

    print(f"\n📕 REFUTING EVIDENCE ({len(evidence.refuting_papers)} papers)")
    print("─" * WIDTH)
    if evidence.refuting_papers:
        for i, paper in enumerate(evidence.refuting_papers, 1):
            _print_paper(paper, i)
    else:
        print("  No refuting papers found.")


def display_synthesis(synthesis: Synthesis) -> None:
    _header("STEP 4 · SYNTHESIS")
    print("\n  Top supporting reasons:")
    for i, p in enumerate(synthesis.supporting_points, 1):
        print(f"    {i}. [{p.strength.upper()}] {p.point}")
    print("\n  Top counter reasons:")
    for i, p in enumerate(synthesis.counter_points, 1):
        print(f"    {i}. [{p.strength.upper()}] {p.point}")
    if synthesis.supporting_summary:
        print("\n  Supporting summary:")
        print(f"    {synthesis.supporting_summary}")
    if synthesis.counter_summary:
        print("\n  Counter summary:")
        print(f"    {synthesis.counter_summary}")


def display_verdict(verdict: Verdict) -> None:
    _header("STEP 5 · VERDICT")
    print(f"\n  Support score: [{_stars(verdict.support_score)}  {verdict.support_score:.2f}]   confidence: {verdict.confidence}")
    if verdict.components:
        print("\n  Claim breakdown:")
        for comp in verdict.components:
            print(f"    • [{comp.status.upper()}] {comp.component}")
            if comp.note:
                print(f"        {comp.note}")
    if verdict.strongest_supported_claim:
        print("\n  Strongest supported version:")
        print(f"    {verdict.strongest_supported_claim}")
    if verdict.unsupported_aspects:
        print("\n  Not supported / untested:")
        for aspect in verdict.unsupported_aspects:
            print(f"    • {aspect}")
    if verdict.reasoning:
        print("\n  Reasoning:")
        print(f"    {verdict.reasoning}")
    print("\n" + "═" * WIDTH + "\n")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


async def run(claim: str, config: PipelineConfig = DEFAULT_CONFIG) -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to .env in the project root.")

    print("\n🔬 Gathering evidence (this may take a couple of minutes):\n")
    result = await gather_evidence(
        claim,
        verbose=config.verbose,
        model_query_gen=config.model_query_gen,
        model_screening=config.model_screening,
        model_extraction=config.model_extraction,
    )

    if config.show_queries:
        display_queries(result)
    if config.show_candidates:
        display_candidates(result)
    if config.show_evidence and result.evidence is not None:
        display_evidence(result.evidence)

    synthesis: Synthesis | None = None
    if config.run_synthesis and result.evidence is not None:
        synthesis = await synthesize(
            result.evidence, verbose=config.verbose, model=config.model_synthesis
        )
        if config.show_synthesis:
            display_synthesis(synthesis)

    if config.run_verdict and synthesis is not None:
        verdict = await generate_verdict(
            claim, synthesis, verbose=config.verbose, model=config.model_verdict
        )
        if config.show_verdict:
            display_verdict(verdict)


def main() -> None:
    print("\n╔══════════════════════════════════╗")
    print("║   crux-lab · claim adjudicator   ║")
    print("╚══════════════════════════════════╝")
    print("\nEnter a scientific claim and I'll gather evidence, synthesize it,")
    print("and render a verdict.\n")
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

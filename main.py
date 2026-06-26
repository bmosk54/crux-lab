"""crux-lab — evidence gatherer for scientific hypotheses."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import anyio
from dotenv import load_dotenv

from agents.evidence_gatherer import gather_evidence
from models.evidence import EvidencePaper, EvidenceSet

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

_STARS = {(0.0, 0.3): "★☆☆☆☆", (0.3, 0.5): "★★☆☆☆", (0.5, 0.7): "★★★☆☆", (0.7, 0.9): "★★★★☆", (0.9, 1.01): "★★★★★"}


def _stars(score: float) -> str:
    for (lo, hi), label in _STARS.items():
        if lo <= score < hi:
            return label
    return "★☆☆☆☆"


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
    width = 72
    print("\n" + "═" * width)
    print(f"  HYPOTHESIS: {evidence.hypothesis}")
    print("═" * width)

    print(f"\n📗 SUPPORTING EVIDENCE ({len(evidence.supporting_papers)} papers)")
    print("─" * width)
    if evidence.supporting_papers:
        for i, paper in enumerate(evidence.supporting_papers, 1):
            _print_paper(paper, i)
    else:
        print("  No supporting papers found.")

    print(f"\n📕 REFUTING EVIDENCE ({len(evidence.refuting_papers)} papers)")
    print("─" * width)
    if evidence.refuting_papers:
        for i, paper in enumerate(evidence.refuting_papers, 1):
            _print_paper(paper, i)
    else:
        print("  No refuting papers found.")

    print("\n" + "═" * width + "\n")


async def run(hypothesis: str) -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to .env in the project root.")

    print("\n🔬 Gathering evidence (this may take 1-2 minutes):\n")
    evidence = await gather_evidence(hypothesis)
    display_evidence(evidence)


def main() -> None:
    print("\n╔══════════════════════════════════╗")
    print("║   crux-lab · evidence gatherer   ║")
    print("╚══════════════════════════════════╝")
    print("\nEnter a scientific hypothesis and I'll find supporting and")
    print("refuting evidence from the biomedical literature.\n")

    hypothesis = input("Hypothesis: ").strip()
    if not hypothesis:
        print("No hypothesis entered. Exiting.")
        sys.exit(0)

    anyio.run(run, hypothesis)


if __name__ == "__main__":
    main()

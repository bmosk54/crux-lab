"""Synthesizer: distil gathered evidence into ranked reasons + summary paragraphs."""

from __future__ import annotations

from agents.llm import complete, parse_tagged_json
from models.evidence import EvidencePaper, EvidenceSet, Synthesis

SYNTH_SYSTEM = """
You are a scientific synthesis analyst. You are given a hypothesis and a structured
set of evidence papers (supporting and refuting). Each paper has claims classified
by evidence_type and directness, plus a citation count and a paper_strength score.

Distil this into the strongest case on each side.

## Rank the reasons
Identify the top 3 SUPPORTING reasons and the top 3 COUNTER reasons (fewer if the
evidence genuinely doesn't support that many). A "reason" is a distinct line of
argument, NOT a single paper — merge papers that make the same point, and order
them strongest first.

## Judge strength (use judgement, not a formula)
Label each reason "strong", "moderate", or "weak" by weighing together:
- evidence_type: meta_analysis/rct > cohort > animal_model/in_vitro > mechanistic/case_report
- directness: direct > indirect > tangential
- consistency & quantity: several independent papers agreeing is stronger
- paper_strength / citations: established, high-impact work counts for more
A lone tangential mechanistic paper is "weak"; converging direct RCT or
meta-analysis evidence is "strong".

## Summaries
Write two cohesive paragraphs:
- supporting_summary: weave the supporting reasons together, explicitly noting how
  strong each is and whether the evidence is direct or indirect.
- counter_summary: do the same for the counter reasons.
Be specific and readable — a knowledgeable reader should understand the state of
the evidence from these paragraphs alone.

## Output format
Output EXACTLY this and nothing after the closing tag:

<synthesis>
{
  "supporting_points": [{"point": "...", "strength": "strong"}],
  "counter_points": [{"point": "...", "strength": "moderate"}],
  "supporting_summary": "...",
  "counter_summary": "..."
}
</synthesis>
""".strip()


def _format_paper(paper: EvidencePaper) -> str:
    year = (paper.published or "?")[:4]
    header = (
        f"- {paper.title} ({year}, {paper.cited_by_count} cit, "
        f"strength {paper.paper_strength:.2f})"
    )
    lines = [header]
    for c in paper.supporting_claims:
        lines.append(f"    [supports | {c.evidence_type} | {c.directness}] {c.claim}")
    for c in paper.refuting_claims:
        lines.append(f"    [refutes  | {c.evidence_type} | {c.directness}] {c.claim}")
    return "\n".join(lines)


def _evidence_to_text(evidence: EvidenceSet) -> str:
    sup = "\n".join(_format_paper(p) for p in evidence.supporting_papers) or "  (none)"
    ref = "\n".join(_format_paper(p) for p in evidence.refuting_papers) or "  (none)"
    return f"SUPPORTING PAPERS:\n{sup}\n\nREFUTING PAPERS:\n{ref}"


async def synthesize(
    evidence: EvidenceSet, *, verbose: bool = True, model: str = "sonnet"
) -> Synthesis:
    if verbose:
        print(f"  → synthesizing ranked reasons and summaries (model: {model})...")

    if not evidence.supporting_papers and not evidence.refuting_papers:
        return Synthesis(
            supporting_summary="No usable evidence was gathered.",
            counter_summary="No usable evidence was gathered.",
        )

    prompt = (
        f"Hypothesis:\n{evidence.hypothesis}\n\n"
        f"Evidence:\n{_evidence_to_text(evidence)}"
    )
    text = await complete(SYNTH_SYSTEM, prompt, model=model)
    data = parse_tagged_json(text, "synthesis")
    if isinstance(data, dict):
        try:
            return Synthesis(**data)
        except (TypeError, ValueError) as exc:
            print(f"[warning] Could not parse synthesis JSON: {exc}")
    return Synthesis()

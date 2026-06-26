"""Verdict: score the claim, decompose it, and state the strongest supported version."""

from __future__ import annotations

from agents.llm import complete, parse_tagged_json
from models.evidence import Synthesis, Verdict

VERDICT_SYSTEM = """
You are a scientific verdict generator. You are given:
- the ORIGINAL claim
- a SYNTHESIS of the evidence: ranked supporting and counter reasons (each with a
  strength label) plus two summary paragraphs.

Judge the claim on the evidence in the synthesis. Do not invent evidence beyond it.

## Score
support_score (0.0–1.0) for the ORIGINAL claim AS STATED:
  1.0 = strongly and directly supported, little credible opposition
  0.5 = genuinely mixed / equivocal
  0.0 = strongly refuted
confidence ("high" / "moderate" / "low") = how much credible evidence exists to
judge at all. Use "low" when evidence is thin, indirect, or sparse, regardless of
which direction it points.

## Decompose
Break the original claim into its distinct sub-components or hidden assumptions
(e.g. an existence part, a magnitude part, a causal/mechanistic part, a scope part
like "in humans"). For each, give a status:
  "supported" / "refuted" / "mixed" / "untested"
with a brief note tied to the evidence.

## Strongest supported claim
Rewrite the claim as the STRONGEST version the evidence ACTUALLY supports: narrow
the scope, add qualifiers, or split it so each part is backed. If the original
claim is already well-calibrated to the evidence, say so and you may restate it.
If almost nothing is strongly supported or refuted, say that plainly rather than
forcing a conclusion.

## Unsupported aspects
List the parts of the original claim the evidence does NOT support or leaves untested.

## Output format
Output EXACTLY this and nothing after the closing tag:

<verdict>
{
  "support_score": 0.0,
  "confidence": "low",
  "components": [{"component": "...", "status": "supported", "note": "..."}],
  "strongest_supported_claim": "...",
  "unsupported_aspects": ["..."],
  "reasoning": "..."
}
</verdict>
""".strip()


def _synthesis_to_text(synthesis: Synthesis) -> str:
    def points(label: str, items) -> str:
        if not items:
            return f"{label}: (none)"
        lines = [f"{label}:"]
        for i, p in enumerate(items, 1):
            lines.append(f"  {i}. [{p.strength}] {p.point}")
        return "\n".join(lines)

    return (
        f"{points('SUPPORTING REASONS', synthesis.supporting_points)}\n\n"
        f"{points('COUNTER REASONS', synthesis.counter_points)}\n\n"
        f"SUPPORTING SUMMARY:\n{synthesis.supporting_summary or '(none)'}\n\n"
        f"COUNTER SUMMARY:\n{synthesis.counter_summary or '(none)'}"
    )


async def generate_verdict(
    claim: str, synthesis: Synthesis, *, verbose: bool = True, model: str = "sonnet"
) -> Verdict:
    if verbose:
        print(f"  → weighing the verdict (model: {model})...")

    prompt = (
        f"Original claim:\n{claim}\n\n"
        f"Synthesis of the evidence:\n{_synthesis_to_text(synthesis)}"
    )
    text = await complete(VERDICT_SYSTEM, prompt, model=model)
    data = parse_tagged_json(text, "verdict")
    if isinstance(data, dict):
        try:
            return Verdict(**data)
        except (TypeError, ValueError) as exc:
            print(f"[warning] Could not parse verdict JSON: {exc}")
    return Verdict(reasoning="Could not generate a verdict from the available evidence.")

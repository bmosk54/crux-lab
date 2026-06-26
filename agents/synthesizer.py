"""Stage 4 — synthesis + verdict (merged into one premium call).

Consumes the original claim, the Stage 0 Decomposition, and the EvidenceSet (each
paper carrying stance, study type, directness, key findings, and full-text section
text where pulled). Emits the structured Verdict.

The reasoning instructions below are the project's verbatim synthesis spec; a JSON
output contract is appended so the prose maps onto machine-readable fields.
"""

from __future__ import annotations

from agents.llm import complete, parse_tagged_json
from models.evidence import Decomposition, EvidencePaper, EvidenceSet, Verdict

# --- Verbatim synthesis spec (do not edit the wording) ---------------------
SYNTH_SPEC = """
You are the synthesis stage of a scientific claim-evaluation system. You receive:
- CLAIM: the original claim, verbatim.
- DECOMPOSITION: atomic sub-claims; load-bearing modifiers; the established
  knowledge the claim engages; the claim type (causal / functional / mechanistic /
  therapeutic / association / paradigm / absence / comparative).
- EVIDENCE: papers, each with stance (support/refute/mixed), study type,
  directness-to-claim, and key extracted findings.

Produce a verdict. Follow these rules exactly.

1. RULE EACH SUB-CLAIM SEPARATELY. Conjunctive claims usually have sub-claims with
   different fates; never average them into one verdict. For each, give a label
   (SUPPORTED / REFUTED / MIXED / UNTESTED) and a one-line basis.

2. RULE ON THE MODIFIER. State the verdict for the claim AS WRITTEN (with its
   load-bearing modifiers) and, if they differ, for the claim WEAKENED (modifiers
   relaxed). If the strong form fails only because of a modifier ("the" root cause,
   "necessary", "reverses"), say so explicitly — that is the most common failure mode.

3. POSITION AGAINST CONSENSUS. State how the claim's subject sits relative to the
   established knowledge it engages: does it sit upstream/downstream of, extend,
   contradict, or duplicate the consensus position? Use the relation appropriate to
   the claim type (e.g. causal ordering for causal claims; canonical-vs-proposed
   function for functional claims; added explanatory scope for paradigm claims;
   strength of existing positive evidence for absence claims). Do NOT refute a
   "primacy/root/first/only" claim merely by listing parallel alternatives — locate
   the claimed entity correctly relative to the established one.

4. WEIGHT COUNTER-EVIDENCE BY APPLICABILITY, NOT PROMINENCE. For each refuting
   paper, first ask: does its mechanism, population, intervention, or measured
   proxy actually bear on THIS claim's mechanism/variables? Down-weight high-profile
   evidence that tests a different thing (e.g. a clinical trial of a surrogate
   marker used against a specific molecular mechanism). Note such mismatches
   explicitly rather than letting citation count drive the verdict.

5. RESPECT CLAIM TYPE WHEN JUDGING SUFFICIENCY. A therapeutic claim is not
   established by mechanism/preclinical evidence alone; a mechanism claim is not
   refuted by a failed surrogate-endpoint trial; a paradigm claim is judged on
   explanatory power and scope, not RCTs; an absence claim cannot be confirmed by a
   null search — report coverage and withhold strong confidence.

6. GENERATE AND SEPARATELY SCORE A STEELMAN. Identify the strongest defensible
   version of the claim that the evidence DOES support — typically by relaxing the
   load-bearing modifier or re-scoping (e.g. "root cause" → "major contributor to
   progression"; "treats X" → "plausible adjunct for the residual problem X-lowering
   leaves"). State it as one sentence and give it its own confidence. If the steelman
   scores materially higher than the literal claim, say so and quantify the gap.

7. NAME THE DECISIVE EXPERIMENT. State the single highest-value-of-information test
   that would most move the verdict — the result that would confirm or break the
   claim — and which direction each outcome points.

8. CALIBRATE. Give a 0–100 confidence per sub-claim and for the overall claim as
   written. Be honest about UNTESTED components: if the evidence speaks to a broader
   category but never isolates the claim's specific entity, label that gap explicitly
   rather than letting category-level evidence stand in for it.

Output: per-sub-claim table → consensus positioning → applicability-adjusted
counter-evidence → steelman (with score) → decisive experiment → overall verdict
with calibrated confidence and an explicit list of what remains untested.
""".strip()

# --- Machine-readable output contract (appended to the verbatim spec) ------
SYNTH_OUTPUT_CONTRACT = """
After reasoning through the rules above, emit your answer as JSON inside a single
<verdict> block and nothing after it. support_score is for the claim AS WRITTEN
(0.0 = strongly refuted, 0.5 = genuinely mixed, 1.0 = strongly supported).

Be DENSE, not verbose — every field is reasoning, not prose. Length limits (hard):
each sub-claim basis ≤ 2 sentences; consensus_positioning ≤ 120 words;
counter_evidence_assessment ≤ 150 words; decisive_experiment ≤ 110 words;
reasoning ≤ 170 words; steelman.statement = ONE sentence. Cite papers as [n].

<verdict>
{
  "sub_claims": [
    {"sub_claim": "...", "label": "supported|refuted|mixed|untested", "confidence": 0, "basis": "..."}
  ],
  "consensus_positioning": "<rule 3: where the claim sits relative to the consensus>",
  "counter_evidence_assessment": "<rule 4: applicability-adjusted, note any mismatches>",
  "steelman": {"statement": "<one sentence>", "confidence": 0},
  "decisive_experiment": "<rule 7: the test + which way each outcome points>",
  "untested_components": ["<components the evidence never isolates>"],
  "support_score": 0.0,
  "overall_label": "supported|refuted|mixed|untested",
  "overall_confidence": 0,
  "reasoning": "<short overall verdict, incl. claim-as-written vs weakened per rule 2>"
}
</verdict>
""".strip()

SYNTH_SYSTEM = f"{SYNTH_SPEC}\n\n{SYNTH_OUTPUT_CONTRACT}"


def _format_evidence_paper(index: int, p: EvidencePaper) -> str:
    year = (p.published or "?")[:4]
    prov = f"{p.authors or 'Unknown'} · {year} · {p.cited_by_count} cit"
    if p.doi:
        prov += f" · doi:{p.doi}"
    head = (
        f"[{index}] stance={p.stance} | study={p.evidence_tier} | directness={p.directness} | "
        f"entity_specific={p.entity_specific} | venue={p.venue_quality}"
        + (" | CONSENSUS" if p.is_consensus_paper else "")
        + (f" | RESERVED({p.reserved_as})" if p.reserved_as else "")
    )
    lines = [head, f"    {p.title}", f"    {prov}"]
    for f in p.key_findings:
        lines.append(f"    - {f}")
    if p.fulltext_sections:
        lines.append(f"    [FULL-TEXT Results/Discussion]\n    {p.fulltext_sections}")
    else:
        lines.append("    [abstract-only]")
    return "\n".join(lines)


def _decomposition_block(d: Decomposition) -> str:
    return (
        f"Sub-claims:\n" + "\n".join(f"  {i}. {s}" for i, s in enumerate(d.sub_claims, 1)) + "\n"
        f"Load-bearing modifiers: {', '.join(d.load_bearing_modifiers) or '(none)'}\n"
        f"Claim type: {d.claim_type}\n"
        f"Established knowledge engaged: {d.established_knowledge or '(none)'}\n"
        f"Named entities (verified): {', '.join(d.named_entities) or '(none)'}"
    )


def _evidence_block(evidence: EvidenceSet) -> str:
    if not evidence.papers:
        return "(no usable evidence was gathered)"
    return "\n\n".join(_format_evidence_paper(i, p) for i, p in enumerate(evidence.papers, 1))


async def synthesize(
    claim: str,
    decomposition: Decomposition,
    evidence: EvidenceSet,
    *,
    verbose: bool = True,
    model: str = "sonnet",
) -> Verdict:
    if verbose:
        print(f"  → synthesizing verdict (model: {model})...")

    prompt = (
        f"CLAIM:\n{claim}\n\n"
        f"DECOMPOSITION:\n{_decomposition_block(decomposition)}\n\n"
        f"EVIDENCE ({len(evidence.papers)} papers):\n{_evidence_block(evidence)}"
    )
    text = await complete(SYNTH_SYSTEM, prompt, model=model)
    data = parse_tagged_json(text, "verdict")

    verdict = Verdict()
    if isinstance(data, dict):
        try:
            verdict = Verdict(**data)
        except (TypeError, ValueError) as exc:
            print(f"[warning] Could not parse verdict JSON: {exc}")
            verdict = Verdict(reasoning="Could not parse a structured verdict from the model output.")

    # Absence claims must not be confirmed from a null/empty search.
    if decomposition.claim_type == "absence":
        n = len(evidence.papers)
        verdict.absence_coverage_note = (
            f"Absence claim: {n} relevant paper(s) retrieved. A null or sparse search "
            "is NOT confirmation — confidence is bounded by search coverage, not asserted."
        )
        if verdict.overall_confidence > 60:
            verdict.overall_confidence = 60

    return verdict

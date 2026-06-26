"""Stage 0 — decomposition. One premium-model call that structures the claim.

Its output (`Decomposition`) is the shared contract every later stage consumes:
queries are driven from it, ranking weights by its claim_type, and the synthesizer
rules on its sub_claims and modifiers.
"""

from __future__ import annotations

from agents.llm import complete, parse_tagged_json
from models.evidence import Decomposition

DECOMPOSE_SYSTEM = """
You are the decomposition stage of a scientific claim-evaluation system. You receive
ONE claim. Break it into a structured object that drives all downstream retrieval and
reasoning. Be claim-type-agnostic: do NOT assume the claim is causal, clinical, or
about any particular field. Read what is actually there.

Emit these fields:

- sub_claims: the atomic, separately-testable assertions packed into the claim. A
  conjunctive claim ("X is the cause AND drug Y fixes it AND it is druggable") has
  several sub-claims that can have different fates. Split them; do not merge.

- load_bearing_modifiers: the exact words/phrases that, if changed, would flip the
  verdict — e.g. "root cause", "necessary", "sufficient", "reverses", "restores",
  "first to show", "only", "no evidence", "cures". These are the words a steelman
  would relax. Each entry is the bare word/phrase ONLY — no explanation or commentary.

- claim_type: EXACTLY one of:
  causal | functional | mechanistic | therapeutic | association | paradigm | absence | comparative

- established_knowledge: the consensus position this claim engages — the canonical
  gene/protein function, the reigning paradigm, the dogma, or (for an absence claim)
  the strongest EXISTING positive evidence the claim denies. End it by stating the
  RELATION the claim has to that consensus, one of:
  extends | contradicts | reorders | duplicates | is-consensus.

- named_entities: candidate specific genes, drugs, programs, methods, assays, or people
  named or strongly implied. These are SEEDS to be VERIFIED by retrieval, not asserted —
  downstream, any entity that returns zero search hits is silently dropped. So it is safe
  to include plausible specific entities; do not omit a specific drug/program/method just
  because you are unsure it exists. Prefer specific names over generic classes.

Rules:
- For claim_type == "absence" (e.g. "there is no evidence that…", "X has no role in…"):
  a null/empty search must NOT be read as confirmation. Note this so the synthesizer
  reports search coverage and withholds strong confidence.
- Keep it tight and faithful to the claim's wording. BUDGET: at most 5 sub_claims,
  at most 10 named_entities (most specific first), and established_knowledge ≤ 90 words.

Output EXACTLY this and nothing else. Emit STRICT, VALID JSON: every array element
is a plain double-quoted string with NO trailing dashes, notes, or commentary. Put
any explanation only inside established_knowledge (a single string).

<decomposition>
{
  "sub_claims": ["atomic claim 1", "atomic claim 2"],
  "load_bearing_modifiers": ["root cause", "will restore"],
  "claim_type": "causal",
  "established_knowledge": "<consensus position>. RELATION: contradicts",
  "named_entities": ["entity 1", "entity 2"]
}
</decomposition>
""".strip()


async def decompose_claim(claim: str, *, model: str = "sonnet", verbose: bool = True) -> Decomposition:
    if verbose:
        print(f"  → decomposing claim (model: {model})...")

    text = await complete(DECOMPOSE_SYSTEM, f"Claim:\n\n{claim}", model=model)
    data = parse_tagged_json(text, "decomposition")
    if isinstance(data, dict):
        data.pop("claim", None)  # never let the LLM override the verbatim claim
        try:
            decomp = Decomposition(**data)
            decomp.claim = claim
            return decomp
        except (TypeError, ValueError) as exc:
            print(f"[warning] Decomposition JSON failed validation: {exc}")
    else:
        print("[warning] Decomposition output was not parseable JSON; using fallback.")

    # Fallback: a minimal decomposition so the pipeline still runs.
    return Decomposition(claim=claim, sub_claims=[claim])

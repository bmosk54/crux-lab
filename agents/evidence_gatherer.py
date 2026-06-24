"""Evidence gatherer agent: finds supporting and refuting literature for a hypothesis."""

from __future__ import annotations

import json
import re

from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ClaudeSDKClient, TextBlock

from models.evidence import EvidencePaper, EvidenceSet, compute_paper_strength
from tools.europepmc import EUROPEPMC_TOOL, europepmc_server

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are a rigorous scientific evidence gatherer. Given a hypothesis, you search
Europe PMC in two phases — a cheap discovery phase and a targeted reading phase —
to find the strongest supporting and refuting papers.

## Phase 1 — Discovery (include_abstracts=false, page_size=12)

Scan titles and citation counts WITHOUT loading abstracts. Run these searches:
  - 2 citation-sorted supporting searches (sort=citedByCount, min_citations=20)
  - 2 citation-sorted refuting searches  (sort=citedByCount, min_citations=20)
  - 1 recent supporting search           (sort=date, min_citations=0)
  - 1 recent refuting search             (sort=date, min_citations=0)

From the ~70 titles you receive, identify your 10-12 best candidates based on:
  - Citation count (prefer highly cited)
  - Title relevance (paper directly about the hypothesis variables)
  - Study design diversity (mix of RCTs, cohorts, animal models, in vitro)

## Phase 2 — Deep reading (include_abstracts=true, page_size=5)

Run 3-4 targeted searches that will surface your shortlisted candidates,
then read their abstracts carefully to extract claims.
Use narrow, specific queries (author names, exact phrases) to find specific papers.

## Selecting your final set

- Choose the top 3-5 SUPPORTING and top 3-5 REFUTING papers.
- STRONGLY prefer papers from citation-sorted passes (min_citations=20).
  Only include a recent (0-citation) paper if it represents a study design
  genuinely absent from the established set (e.g. the only RCT, or a direct
  replication failure).
- Favour DIVERSITY of study design over multiple papers making the same point.
- A paper may appear in both lists if its abstract contains genuinely mixed evidence.

## Classifying each claim

evidence_type — the methodological category of the study:
  "rct"          — Randomised controlled trial
  "meta_analysis"— Systematic review or meta-analysis
  "cohort"       — Cohort, case-control, or large observational study
  "animal_model" — In vivo animal experiment
  "in_vitro"     — Cell culture or ex vivo experiment
  "mechanistic"  — Proposed mechanism or theoretical framework (no direct test)
  "case_report"  — Case report or case series

directness — does the abstract's PRIMARY measurement match the hypothesis variable:
  "direct"    — The paper directly measured the EXACT variable(s) stated in the
                hypothesis (e.g. hypothesis = "X declines with Y" → paper measured
                X in Y conditions, regardless of the study's original purpose)
  "indirect"  — The paper measured something causally one step from the hypothesis
                variable (e.g. upstream/downstream correlate), requiring one
                inferential step to connect to the hypothesis
  "tangential"— Connection requires multiple reasoning steps, or the biological
                context differs substantially (different organism, disease vs aging)

## Output format

After both phases are complete, output EXACTLY this — nothing after the closing tag:

<evidence>
{
  "hypothesis": "<hypothesis exactly as given>",
  "supporting_papers": [
    {
      "title": "...",
      "authors": "...",
      "journal": "...",
      "published": "YYYY-MM-DD",
      "pmid": "..." or null,
      "pmcid": "..." or null,
      "doi": "..." or null,
      "cited_by_count": 0,
      "abstract_snippet": "<most relevant 1-3 sentences from the abstract>",
      "supporting_claims": [
        {"claim": "...", "evidence_type": "rct", "directness": "direct"}
      ],
      "refuting_claims": [
        {"claim": "...", "evidence_type": "cohort", "directness": "indirect"}
      ]
    }
  ],
  "refuting_papers": [ ... same schema ... ]
}
</evidence>
""".strip()

# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------


async def gather_evidence(hypothesis: str) -> EvidenceSet:
    """Run the evidence-gathering agent and return a validated EvidenceSet."""
    options = ClaudeAgentOptions(
        model="sonnet",
        system_prompt=SYSTEM_PROMPT,
        max_turns=14,
        mcp_servers={"europepmc": europepmc_server},
        allowed_tools=[EUROPEPMC_TOOL],
    )

    full_text = ""
    async with ClaudeSDKClient(options=options) as client:
        await client.query(f"Hypothesis to investigate:\n\n{hypothesis}")
        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        full_text += block.text

    evidence = _parse_evidence(full_text, hypothesis)

    # Compute paper_strength in Python so it's consistent and deterministic
    for paper in evidence.supporting_papers + evidence.refuting_papers:
        paper.paper_strength = compute_paper_strength(paper.cited_by_count, paper.published)

    return evidence


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_evidence(text: str, hypothesis: str) -> EvidenceSet:
    """Extract the JSON block from the agent's response and parse it."""
    # Primary: look for <evidence>...</evidence>
    match = re.search(r"<evidence>\s*(.*?)\s*</evidence>", text, re.DOTALL)
    if not match:
        # Fallback: bare JSON code block
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)

    if match:
        raw = match.group(1).strip()
        try:
            data = json.loads(raw)
            return EvidenceSet(**data)
        except (json.JSONDecodeError, Exception) as exc:
            print(f"[warning] Could not parse evidence JSON: {exc}")

    return EvidenceSet(hypothesis=hypothesis)

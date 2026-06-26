"""Pipeline configuration: toggle which stages run and which step outputs print.

Flip any flag here (or construct a PipelineConfig in code) to turn a stage or its
debug output on/off independently. Stages run in order, so disabling an earlier
stage that a later one depends on will also skip the dependent stage.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PipelineConfig:
    # --- which stages to run ---
    run_synthesis: bool = True   # gathering always runs; this gates synthesis
    run_verdict: bool = True     # requires synthesis

    # --- which step outputs to print (debugging visibility) ---
    show_queries: bool = True
    show_candidates: bool = True
    show_evidence: bool = True
    show_synthesis: bool = True
    show_verdict: bool = True

    # --- progress logging (the "→ ..." lines) ---
    verbose: bool = True


# Default config used by main.py. Edit these to quickly turn steps on/off.
DEFAULT_CONFIG = PipelineConfig()


# Diverse, interesting claims for testing the full pipeline. Each is chosen to
# stress a different behaviour (mixed evidence, decomposable claims, strong vs.
# contested causation, animal-vs-human gaps, etc.).
EXAMPLE_CLAIMS = [
    # Strong, well-replicated causal claim — should score high with confidence.
    "SGLT2 inhibitors reduce cardiovascular mortality in patients with heart failure.",
    # Decomposable: 'shortening occurs' (supported) vs 'is a primary cause' (contested).
    "Telomere shortening is a primary cause of human aging.",
    # Classic mixed/contested RCT literature — good for an equivocal verdict.
    "Vitamin D supplementation reduces the incidence of acute respiratory infections.",
    # Strong in animals, thin/indirect in humans — tests species-gap reasoning.
    "Intermittent fasting extends lifespan in mammals.",
    # Emerging mechanistic/animal field — should land low-confidence.
    "The gut microbiome causally influences anxiety-related behavior.",
    # Popular belief vs. evidence — tests refutation handling.
    "Dietary sugar intake causes hyperactivity in children.",
    # Overstated scope — strongest-supported-claim should narrow it.
    "Omega-3 fatty acid supplementation prevents cognitive decline in older adults.",
]

"""Shared single-shot Claude completion + JSON parsing helpers for crux-lab agents."""

from __future__ import annotations

import json
import re
from typing import Any

from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, TextBlock, query

MODEL = "sonnet"


async def complete(system: str, prompt: str, *, model: str = MODEL) -> str:
    """Run a single-shot Claude completion (no tools) and return its text."""
    options = ClaudeAgentOptions(
        model=model,
        system_prompt=system,
        max_turns=1,
        allowed_tools=[],
    )
    out = ""
    async for msg in query(prompt=prompt, options=options):
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    out += block.text
    return out


def parse_tagged_json(text: str, tag: str) -> Any | None:
    """Extract JSON from a <tag>...</tag> block, falling back to a ```json fence."""
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL)
    if not match:
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return None

"""Shared single-shot Claude completion + JSON parsing helpers for crux-lab agents."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, TextBlock, query

MODEL = "sonnet"


async def _complete_once(system: str, prompt: str, *, model: str) -> str:
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


async def complete(system: str, prompt: str, *, model: str = MODEL, retries: int = 1) -> str:
    """Single-shot Claude completion (no tools) with a small retry for transient
    SDK errors (the agent SDK intermittently raises e.g. "error result: success"
    or "Reached maximum number of turns"). Returns the assistant text."""
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return await _complete_once(system, prompt, model=model)
        except Exception as exc:  # noqa: BLE001 — SDK raises bare Exception
            last_exc = exc
            if attempt < retries:
                await asyncio.sleep(1.5)
    raise last_exc  # type: ignore[misc]


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

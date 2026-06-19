"""Starter template for prompting Claude via claude-agent-sdk."""

import os
from pathlib import Path

import anyio
from dotenv import load_dotenv

from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, TextBlock, query

# Load API key from .env in the project root
load_dotenv(Path(__file__).resolve().parent / ".env")

INITIAL_PROMPT = """
You are helping me build a project called crux_bio.

For this first message, briefly introduce yourself and confirm you can read this prompt.
Then ask what I want to work on first.
""".strip()


async def run_prompt(prompt: str = INITIAL_PROMPT) -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to .env in the project root."
        )

    options = ClaudeAgentOptions(
        model="haiku",
        cwd=str(Path(__file__).resolve().parent),
        max_turns=1,
    )

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(block.text)


if __name__ == "__main__":
    anyio.run(run_prompt)

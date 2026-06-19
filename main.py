"""crux-lab — playful sandbox for learning agentic tooling."""

import os
from pathlib import Path

import anyio
from dotenv import load_dotenv

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    TextBlock,
)
from tools.europepmc import EUROPEPMC_TOOL, europepmc_server

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

INITIAL_PROMPT = """
You are helping me in crux-lab — my bio-flavored playground for learning
agentic programming and building with AI agents.

You have access to a Europe PMC literature search tool. When I ask about
papers, genes, or biomedical topics, use that tool before answering.

Briefly introduce yourself, confirm you can read this prompt, and tell me
you can search Europe PMC. Then ask what experiment I want to run first.
""".strip()

DEMO_PROMPT = (
    "Search Europe PMC for recent papers involving bioelectricity as it relates to regeneration and summarize the top 3 results."
)

def build_options() -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        model="haiku",
        cwd=str(ROOT),
        max_turns=5,
        mcp_servers={"europepmc": europepmc_server},
        allowed_tools=[EUROPEPMC_TOOL],
    )


async def run_prompt(prompt: str = INITIAL_PROMPT) -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to .env in the project root."
        )

    options = build_options()

    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)

        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(block.text)


if __name__ == "__main__":
    import sys

    prompt = DEMO_PROMPT if len(sys.argv) > 1 and sys.argv[1] == "--demo" else INITIAL_PROMPT
    anyio.run(run_prompt, prompt)

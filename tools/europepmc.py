"""Europe PMC literature search tool for crux-lab agents."""

from __future__ import annotations

import json
from typing import Any

import httpx
from claude_agent_sdk import create_sdk_mcp_server, tool

EUROPEPMC_SEARCH_URL = (
    "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
)
MAX_PAGE_SIZE = 25

SEARCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": (
                "Europe PMC search query. Examples: 'CRISPR', "
                "'auth:smith', 'journal:nature', '\"gene therapy\"'"
            ),
        },
        "page_size": {
            "type": "integer",
            "description": f"Number of results to return (1-{MAX_PAGE_SIZE}).",
            "default": 10,
        },
    },
    "required": ["query"],
}


def _format_hit(hit: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": hit.get("id"),
        "source": hit.get("source"),
        "title": hit.get("title"),
        "authors": hit.get("authorString"),
        "journal": hit.get("journalTitle"),
        "published": hit.get("firstPublicationDate"),
        "pmid": hit.get("pmid"),
        "pmcid": hit.get("pmcid"),
        "doi": hit.get("doi"),
        "is_open_access": hit.get("isOpenAccess"),
        "cited_by_count": hit.get("citedByCount"),
    }


@tool(
    "search_literature",
    "Search Europe PMC for life-science publications and preprints.",
    SEARCH_SCHEMA,
)
async def search_literature(args: dict[str, Any]) -> dict[str, Any]:
    query = args["query"].strip()
    if not query:
        return {
            "content": [{"type": "text", "text": "Error: query must not be empty."}],
            "is_error": True,
        }

    page_size = min(max(int(args.get("page_size", 10)), 1), MAX_PAGE_SIZE)

    params = {
        "query": query,
        "format": "json",
        "resultType": "lite",
        "pageSize": page_size,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(EUROPEPMC_SEARCH_URL, params=params)
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPError as exc:
        return {
            "content": [
                {"type": "text", "text": f"Europe PMC request failed: {exc}"}
            ],
            "is_error": True,
        }

    results = payload.get("resultList", {}).get("result", [])
    summary = {
        "query": query,
        "total_hits": payload.get("hitCount", 0),
        "returned": len(results),
        "results": [_format_hit(hit) for hit in results],
    }

    return {
        "content": [
            {"type": "text", "text": json.dumps(summary, indent=2)},
        ],
    }


europepmc_server = create_sdk_mcp_server(
    name="europepmc",
    version="1.0.0",
    tools=[search_literature],
)

# Tool name Claude uses when calling through the SDK MCP server.
EUROPEPMC_TOOL = "mcp__europepmc__search_literature"

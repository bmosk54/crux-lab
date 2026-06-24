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
                "Europe PMC search query. Supports field syntax: "
                "'auth:smith', 'journal:nature', '\"exact phrase\"', "
                "'MESH:D015967'. Do NOT include citation filters here — "
                "use min_citations instead."
            ),
        },
        "page_size": {
            "type": "integer",
            "description": f"Number of results to return (1-{MAX_PAGE_SIZE}).",
            "default": 10,
        },
        "sort": {
            "type": "string",
            "enum": ["relevance", "citedByCount", "date"],
            "description": (
                "Sort order. Use 'citedByCount' for established/high-impact papers, "
                "'date' for the most recent papers, 'relevance' (default) for "
                "broad topic searches."
            ),
            "default": "relevance",
        },
        "min_citations": {
            "type": "integer",
            "description": (
                "Minimum citation count. Set to 20+ for established literature passes, "
                "0 for recent-paper passes where low citation count is expected."
            ),
            "default": 0,
        },
        "include_abstracts": {
            "type": "boolean",
            "description": (
                "Whether to return full abstracts. "
                "Use false (default) for Phase 1 discovery — surveys titles and "
                "citations cheaply across many results. "
                "Use true only for Phase 2 deep-reading of your shortlisted candidates."
            ),
            "default": False,
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
        "abstract": hit.get("abstractText"),
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
    sort = args.get("sort", "relevance")
    min_citations = max(0, int(args.get("min_citations", 0)))
    include_abstracts = bool(args.get("include_abstracts", False))

    if min_citations > 0:
        query = f"{query} (CITED_BY_COUNT:[{min_citations} TO *])"

    sort_map = {"citedByCount": "citedByCount:desc", "date": "P_PDATE_D:desc"}

    params: dict[str, Any] = {
        "query": query,
        "format": "json",
        "resultType": "core" if include_abstracts else "lite",
        "pageSize": page_size,
    }
    if sort in sort_map:
        params["sort"] = sort_map[sort]

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

EUROPEPMC_TOOL = "mcp__europepmc__search_literature"

"""Europe PMC literature search tool for crux-lab agents.

Exposes both:
- An in-process MCP tool (`search_literature`) for agentic use.
- Plain async helpers (`run_search`, `fetch_abstracts`) for Python-orchestrated
  pipelines that need deterministic control over retrieval.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
from claude_agent_sdk import create_sdk_mcp_server, tool

EUROPEPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
MAX_PAGE_SIZE = 25
_SORT_MAP = {"citedByCount": "CITED desc", "date": "P_PDATE_D desc"}

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
                "Whether to return full abstracts. Use false (default) for cheap "
                "discovery over titles; true to deep-read shortlisted candidates."
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


async def run_search(
    query: str,
    *,
    page_size: int = 10,
    sort: str = "relevance",
    min_citations: int = 0,
    include_abstracts: bool = False,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Run a single Europe PMC search and return a structured summary.

    Result order is preserved so callers can use position as a relevance signal.
    """
    query = query.strip()
    if not query:
        return {"query": query, "total_hits": 0, "returned": 0, "results": []}

    page_size = min(max(int(page_size), 1), MAX_PAGE_SIZE)
    min_citations = max(0, int(min_citations))
    if min_citations > 0:
        query = f"{query} AND CITED:[{min_citations} TO *]"

    params: dict[str, Any] = {
        "query": query,
        "format": "json",
        "resultType": "core" if include_abstracts else "lite",
        "pageSize": page_size,
    }
    if sort in _SORT_MAP:
        params["sort"] = _SORT_MAP[sort]

    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(timeout=30.0)
    try:
        response = await client.get(EUROPEPMC_SEARCH_URL, params=params)
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPError as exc:
        return {"query": query, "error": str(exc), "total_hits": 0, "returned": 0, "results": []}
    finally:
        if owns_client:
            await client.aclose()

    results = payload.get("resultList", {}).get("result", [])
    return {
        "query": query,
        "total_hits": payload.get("hitCount", 0),
        "returned": len(results),
        "results": [_format_hit(hit) for hit in results],
    }


async def _fetch_one(client: httpx.AsyncClient, source: str, ext_id: str) -> dict[str, Any] | None:
    """Fetch the full record (with abstract) for one specific paper by ID."""
    params = {
        "query": f"ext_id:{ext_id} src:{source}",
        "format": "json",
        "resultType": "core",
        "pageSize": 1,
    }
    try:
        response = await client.get(EUROPEPMC_SEARCH_URL, params=params)
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPError:
        return None
    results = payload.get("resultList", {}).get("result", [])
    return _format_hit(results[0]) if results else None


async def fetch_abstracts(refs: list[tuple[str, str]]) -> dict[str, dict[str, Any]]:
    """Fetch full records (with abstracts) for specific (source, ext_id) pairs.

    Returns a dict keyed by ext_id. Requests run concurrently.
    """
    if not refs:
        return {}
    async with httpx.AsyncClient(timeout=30.0) as client:
        fetched = await asyncio.gather(
            *(_fetch_one(client, source, ext_id) for source, ext_id in refs),
            return_exceptions=True,
        )
    out: dict[str, dict[str, Any]] = {}
    for (_source, ext_id), result in zip(refs, fetched):
        if isinstance(result, dict) and result:
            out[ext_id] = result
    return out


@tool(
    "search_literature",
    "Search Europe PMC for life-science publications and preprints.",
    SEARCH_SCHEMA,
)
async def search_literature(args: dict[str, Any]) -> dict[str, Any]:
    if not args.get("query", "").strip():
        return {
            "content": [{"type": "text", "text": "Error: query must not be empty."}],
            "is_error": True,
        }

    summary = await run_search(
        args["query"],
        page_size=args.get("page_size", 10),
        sort=args.get("sort", "relevance"),
        min_citations=args.get("min_citations", 0),
        include_abstracts=bool(args.get("include_abstracts", False)),
    )

    return {"content": [{"type": "text", "text": json.dumps(summary, indent=2)}]}


europepmc_server = create_sdk_mcp_server(
    name="europepmc",
    version="1.0.0",
    tools=[search_literature],
)

EUROPEPMC_TOOL = "mcp__europepmc__search_literature"

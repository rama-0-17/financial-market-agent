"""
Web search tool with automatic fallback.

Primary:  Tavily (structured search, great for financial news)
Fallback: DuckDuckGo via requests (no API key needed)

This is a real example of error recovery — if Tavily fails or
the key isn't set, we transparently fall back to DDG.
"""
from __future__ import annotations
import os
import json
import time
import requests
from langchain_core.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


# ── DuckDuckGo fallback (no API key) ─────────────────────────────────────────

def _ddg_search(query: str, max_results: int = 5) -> list[dict]:
    """
    DuckDuckGo instant answer API — free, no key, rate-limited.
    Falls back gracefully to an empty list on failure.
    """
    try:
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1,
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        results = []
        # Abstract (Wikipedia-style summary)
        if data.get("AbstractText"):
            results.append({
                "title": data.get("Heading", "Summary"),
                "content": data["AbstractText"],
                "url": data.get("AbstractURL", ""),
                "source": "DuckDuckGo Abstract",
            })

        # Related topics
        for topic in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append({
                    "title": topic.get("Text", "")[:80],
                    "content": topic.get("Text", ""),
                    "url": topic.get("FirstURL", ""),
                    "source": "DuckDuckGo",
                })

        return results[:max_results]
    except Exception as e:
        return [{"error": f"DuckDuckGo search failed: {e}", "source": "DDG-fallback"}]


# ── Tavily primary ────────────────────────────────────────────────────────────

@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    reraise=False,
)
def _tavily_search(query: str, max_results: int = 5) -> list[dict]:
    """Search via Tavily with retry on transient errors."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY not set")

    from tavily import TavilyClient
    client = TavilyClient(api_key=api_key)
    response = client.search(
        query=query,
        max_results=max_results,
        search_depth="advanced",
        include_answer=True,
    )
    results = []
    if response.get("answer"):
        results.append({
            "title": "AI Answer",
            "content": response["answer"],
            "url": "",
            "source": "Tavily Answer",
        })
    for r in response.get("results", []):
        results.append({
            "title": r.get("title", ""),
            "content": r.get("content", ""),
            "url": r.get("url", ""),
            "score": r.get("score", 0),
            "source": "Tavily",
        })
    return results


# ── LangChain Tool ────────────────────────────────────────────────────────────

@tool
def web_search_tool(query: str, max_results: int = 5) -> str:
    """
    Search the web for current financial news, analyst reports, and company information.
    Automatically falls back from Tavily to DuckDuckGo if Tavily is unavailable.

    Args:
        query: Search query string
        max_results: Maximum number of results to return (default 5)

    Returns:
        JSON string with search results including title, content, and URL
    """
    results = []
    source_used = "none"

    # Try Tavily first
    try:
        results = _tavily_search(query, max_results)
        source_used = "tavily"
    except Exception as tavily_err:
        # Transparent fallback to DuckDuckGo
        results = _ddg_search(query, max_results)
        source_used = "duckduckgo-fallback"
        results.append({
            "note": f"Tavily unavailable ({tavily_err}), used DuckDuckGo fallback",
            "source": "system",
        })

    return json.dumps({
        "query": query,
        "source": source_used,
        "result_count": len(results),
        "results": results,
    }, indent=2)


@tool
def news_search_tool(company: str, days_back: int = 30) -> str:
    """
    Search for recent news about a company.

    Args:
        company: Company name or ticker symbol
        days_back: How many days back to search (default 30)

    Returns:
        JSON string with recent news articles
    """
    query = f"{company} financial news earnings analyst 2025"
    return web_search_tool.invoke({"query": query, "max_results": 6})

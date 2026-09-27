from __future__ import annotations

import html as html_lib
import re
import urllib.error
import urllib.parse
import urllib.request

from microcode.tooling import ToolDefinition, ToolResult

SEARCH_URL = "https://html.duckduckgo.com/html/"
DEFAULT_RESULTS = 5
MAX_RESULTS = 10
SEARCH_TIMEOUT = 15
MAX_BYTES = 200_000
USER_AGENT = "MicroCode/0.1 (local coding agent)"


def _validate(input_data: dict) -> dict:
    query = input_data.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query is required")
    raw = input_data.get("num_results", DEFAULT_RESULTS)
    if isinstance(raw, bool):
        raise ValueError("num_results must be an integer")
    if isinstance(raw, float) and raw.is_integer():
        raw = int(raw)
    if not isinstance(raw, int) or raw < 1 or raw > MAX_RESULTS:
        raise ValueError(f"num_results must be between 1 and {MAX_RESULTS}")
    return {"query": query.strip(), "num_results": raw}


def _strip_tags(text: str) -> str:
    cleaned = re.sub(r"<[^>]+>", "", text)
    return html_lib.unescape(re.sub(r"\s+", " ", cleaned)).strip()


def _unwrap_result_url(href: str) -> str:
    parsed = urllib.parse.urlparse(urllib.parse.urljoin(SEARCH_URL, href))
    query = urllib.parse.parse_qs(parsed.query)
    if "uddg" in query and query["uddg"][0]:
        return query["uddg"][0]
    return href


def parse_search_results(html: str, max_results: int) -> list[dict[str, str]]:
    pattern = re.compile(
        r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?'
        r'(?:<a[^>]*class="result__snippet"[^>]*>(.*?)</a>|<div[^>]*class="result__snippet"[^>]*>(.*?)</div>)',
        re.DOTALL,
    )
    results: list[dict[str, str]] = []
    for match in pattern.finditer(html):
        if len(results) >= max_results:
            break
        title = _strip_tags(match.group(2))
        snippet = _strip_tags(match.group(3) or match.group(4) or "")
        url = _unwrap_result_url(html_lib.unescape(match.group(1)))
        if url and title:
            results.append({"title": title, "url": url, "snippet": snippet[:200]})
    return results


def _download_search_page(query: str) -> str:
    url = f"{SEARCH_URL}?q={urllib.parse.quote(query)}"
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8",
        },
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=SEARCH_TIMEOUT) as response:
        raw = response.read(MAX_BYTES + 1)
    return raw[:MAX_BYTES].decode("utf-8", errors="replace")


def _format_results(query: str, results: list[dict[str, str]]) -> str:
    lines = [f"Search results for: {query}", ""]
    for index, item in enumerate(results, start=1):
        lines.append(f"{index}. {item['title']}")
        lines.append(f"   {item['url']}")
        if item["snippet"]:
            lines.append(f"   {item['snippet']}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _run(input_data: dict, context) -> ToolResult:
    query = input_data["query"]
    if not context.approve(f"Search the web?\n{query}", kind="command", key=f"web_search {query}"):
        return ToolResult(ok=False, output=context.reject_text(f"search: {query}"))

    try:
        page = _download_search_page(query)
    except urllib.error.HTTPError as error:
        return ToolResult(ok=False, output=f"HTTP {error.code}: {error.reason}\nQuery: {query}")
    except urllib.error.URLError as error:
        return ToolResult(ok=False, output=f"Search failed: {error.reason}\nQuery: {query}")
    except OSError as error:
        return ToolResult(ok=False, output=f"Search failed: {error}\nQuery: {query}")

    results = parse_search_results(page, input_data["num_results"])
    if not results:
        return ToolResult(ok=False, output=f"No search results for: {query}")
    return ToolResult(ok=True, output=_format_results(query, results))


web_search_tool = ToolDefinition(
    name="web_search",
    description=(
        "Search the public web and return titles, URLs, and short snippets. "
        "No API key required. Use web_fetch to read a specific result URL. "
        "Prefer grep_files and read_file for questions about this repository."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "num_results": {"type": "integer"},
        },
        "required": ["query"],
    },
    validator=_validate,
    run=_run,
)

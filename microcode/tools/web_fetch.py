from __future__ import annotations

import ipaddress
import re
import socket
import urllib.error
import urllib.request
from urllib.parse import urlparse

from microcode.tooling import ToolDefinition, ToolResult

DEFAULT_MAX_CHARS = 10000
MAX_CHARS = 50000
FETCH_TIMEOUT = 20
MAX_BYTES = 200_000


def _validate(input_data: dict) -> dict:
    url = input_data.get("url")
    if not isinstance(url, str) or not url.strip():
        raise ValueError("url is required")
    max_chars = int(input_data.get("max_chars", DEFAULT_MAX_CHARS))
    if max_chars < 100 or max_chars > MAX_CHARS:
        raise ValueError(f"max_chars must be between 100 and {MAX_CHARS}")
    return {"url": url.strip(), "max_chars": max_chars}


def url_is_allowed(url: str) -> tuple[bool, str]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False, "url must start with http:// or https://"
    host = parsed.hostname
    if not host:
        return False, "url has no hostname"
    if host.lower() in {"localhost"}:
        return False, f"blocked host: {host}"
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror:
        return False, f"could not resolve host: {host}"
    for info in infos:
        raw_ip = info[4][0]
        try:
            ip = ipaddress.ip_address(raw_ip)
        except ValueError:
            continue
        if (
            ip.is_loopback
            or ip.is_private
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return False, f"blocked address for host {host}"
    return True, "ok"


def _run(input_data: dict, context) -> ToolResult:
    url = input_data["url"]
    allowed, reason = url_is_allowed(url)
    if not allowed:
        return ToolResult(ok=False, output=reason)

    if not context.approve(f"Fetch URL?\n{url}", kind="command", key=f"web_fetch {url}"):
        return ToolResult(ok=False, output=f"User rejected fetch: {url}")

    try:
        text, content_type, status = _download(url)
    except urllib.error.HTTPError as error:
        return ToolResult(ok=False, output=f"HTTP {error.code}: {error.reason}\nURL: {url}")
    except urllib.error.URLError as error:
        return ToolResult(ok=False, output=f"Fetch failed: {error.reason}\nURL: {url}")
    except PermissionError as error:
        return ToolResult(ok=False, output=str(error))
    except OSError as error:
        return ToolResult(ok=False, output=f"Fetch failed: {error}\nURL: {url}")

    if "html" in content_type.lower():
        text = _html_to_text(text)
    truncated = len(text) > input_data["max_chars"]
    if truncated:
        text = text[: input_data["max_chars"]] + "\n... (truncated)"
    header = (
        f"URL: {url}\n"
        f"STATUS: {status}\n"
        f"CONTENT_TYPE: {content_type}\n"
        f"TRUNCATED: {'yes' if truncated else 'no'}\n\n"
    )
    return ToolResult(ok=True, output=header + text)


def _download(url: str) -> tuple[str, str, int]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "MicroCode/0.1 (local coding agent)",
            "Accept": "text/html,application/xhtml+xml,text/plain,application/json;q=0.9,*/*;q=0.8",
        },
        method="GET",
    )
    opener = urllib.request.build_opener(_NoRedirectHandler())
    with opener.open(request, timeout=FETCH_TIMEOUT) as response:
        content_type = response.headers.get("Content-Type", "application/octet-stream")
        status = getattr(response, "status", 200)
        raw = response.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raw = raw[:MAX_BYTES]
        charset = "utf-8"
        if "charset=" in content_type:
            charset = content_type.split("charset=", 1)[1].split(";", 1)[0].strip() or "utf-8"
        try:
            text = raw.decode(charset, errors="replace")
        except LookupError:
            text = raw.decode("utf-8", errors="replace")
        return text, content_type, status


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _html_to_text(html: str) -> str:
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
    )
    return re.sub(r"\s+", " ", text).strip()


web_fetch_tool = ToolDefinition(
    name="web_fetch",
    description=(
        "Fetch a public http(s) URL and return text (HTML stripped to readable text). "
        "Use this to read documentation or API pages. Local and private addresses are blocked."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "max_chars": {"type": "number"},
        },
        "required": ["url"],
    },
    validator=_validate,
    run=_run,
)

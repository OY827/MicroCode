from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any


def post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    *,
    timeout: int = 120,
    max_retries: int = 2,
) -> dict[str, Any]:
    """POST JSON and return the decoded object. Retries 429/5xx."""

    body = json.dumps(payload).encode("utf-8")
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        request = urllib.request.Request(url=url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                return _read_json(response.read(), getattr(response, "status", 200))
        except urllib.error.HTTPError as error:
            raw = error.read()
            if _should_retry(error.code) and attempt < max_retries:
                last_error = error
                time.sleep(1.5 * (attempt + 1))
                continue
            data = _decode_json(raw)
            raise RuntimeError(_error_message(data, error.code, raw)) from error
        except urllib.error.URLError as error:
            last_error = error
            if attempt >= max_retries:
                raise RuntimeError(f"Model API connection failed: {error}") from error
            time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"Model API request failed: {last_error}")


def post_sse(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    *,
    timeout: int = 120,
    max_retries: int = 2,
) -> Iterator[dict[str, Any]]:
    """POST JSON and yield SSE (or a single JSON body) as it arrives.

    Retries 429/5xx only when no event has been yielded yet.
    """

    body = json.dumps(payload).encode("utf-8")
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        request = urllib.request.Request(url=url, data=body, headers=headers, method="POST")
        yielded = False
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                for event in iter_sse_or_json(response):
                    yielded = True
                    yield event
                return
        except urllib.error.HTTPError as error:
            raw = error.read()
            if yielded:
                raise RuntimeError(_error_message(_decode_json(raw), error.code, raw)) from error
            if _should_retry(error.code) and attempt < max_retries:
                last_error = error
                time.sleep(1.5 * (attempt + 1))
                continue
            data = _decode_json(raw)
            raise RuntimeError(_error_message(data, error.code, raw)) from error
        except urllib.error.URLError as error:
            if yielded:
                raise RuntimeError(f"Model API connection failed: {error}") from error
            last_error = error
            if attempt >= max_retries:
                raise RuntimeError(f"Model API connection failed: {error}") from error
            time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"Model API request failed: {last_error}")


def iter_sse_or_json(response: Any) -> Iterator[dict[str, Any]]:
    """Read a urllib-style response as SSE events, or one JSON object."""

    first = response.readline()
    if not first:
        return
    first_text = first.decode("utf-8", errors="replace")
    if first_text.lstrip().startswith("{"):
        rest = response.read()
        rest_text = rest.decode("utf-8", errors="replace") if isinstance(rest, bytes) else str(rest)
        data = _decode_json((first_text + rest_text).encode("utf-8"))
        if data:
            yield data
        return
    yield from iter_sse_events(_line_source(response, first_text))


def iter_sse_events(lines: Iterator[str]) -> Iterator[dict[str, Any]]:
    """Parse Server-Sent Events. Ignores event names; uses JSON data objects."""

    data_lines: list[str] = []
    for raw_line in lines:
        line = raw_line.rstrip("\r")
        if line == "":
            event = _sse_data_event(data_lines)
            data_lines = []
            if event is not None:
                yield event
            continue
        if line.startswith(":") or line.startswith("event:") or line.startswith("id:") or line.startswith("retry:"):
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    event = _sse_data_event(data_lines)
    if event is not None:
        yield event


def _line_source(response: Any, first_text: str) -> Iterator[str]:
    yield first_text.rstrip("\r\n")
    while True:
        raw = response.readline()
        if not raw:
            break
        yield raw.decode("utf-8", errors="replace").rstrip("\r\n")


def _sse_data_event(data_lines: list[str]) -> dict[str, Any] | None:
    if not data_lines:
        return None
    payload = "\n".join(data_lines).strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _should_retry(status: int) -> bool:
    return status == 429 or 500 <= status < 600


def _decode_json(raw: bytes) -> dict[str, Any]:
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"error": {"message": text}}
    return parsed if isinstance(parsed, dict) else {"error": {"message": text}}


def _read_json(raw: bytes, status: int) -> dict[str, Any]:
    data = _decode_json(raw)
    if status >= 400:
        raise RuntimeError(_error_message(data, status, raw))
    return data


def _error_message(data: dict[str, Any], status: int, raw: bytes) -> str:
    error = data.get("error")
    if isinstance(error, dict) and error.get("message"):
        return f"Model API error {status}: {error['message']}"
    if isinstance(error, str) and error:
        return f"Model API error {status}: {error}"
    text = raw.decode("utf-8", errors="replace").strip()
    return f"Model API error {status}: {text or 'unknown error'}"

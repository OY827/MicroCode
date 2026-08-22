from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
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

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _nonempty(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def load_dotenv(path: Path | None = None) -> Path | None:
    """Load KEY=VALUE pairs from `.env`. Fills missing or blank keys; does not replace non-empty env vars."""

    candidates = [path] if path is not None else [Path.cwd() / ".env", _project_root() / ".env"]
    env_path = next((candidate for candidate in candidates if candidate is not None and candidate.is_file()), None)
    if env_path is None:
        return None

    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and not _nonempty(key):
            os.environ[key] = value
    return env_path


@dataclass(slots=True, frozen=True)
class ModelConfig:
    api_key: str
    base_url: str
    model: str
    max_tokens: int = 8192
    protocol: str = "auto"

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key.strip())


def resolve_protocol(config: ModelConfig) -> str:
    """Return 'anthropic' or 'openai'. URLs containing 'anthropic' use Messages API."""

    explicit = config.protocol.strip().lower()
    if explicit in {"anthropic", "openai"}:
        return explicit
    if "anthropic" in config.base_url.lower():
        return "anthropic"
    return "openai"


@dataclass(slots=True, frozen=True)
class FallbackConfig:
    api_key: str = ""
    base_url: str = ""
    model: str = ""

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key.strip())


def load_fallback_config() -> FallbackConfig:
    """Optional spare model used only when the primary key is missing or invalid."""

    return FallbackConfig(
        api_key=_nonempty("MICROCODE_FALLBACK_API_KEY"),
        base_url=_nonempty("MICROCODE_FALLBACK_BASE_URL"),
        model=_nonempty("MICROCODE_FALLBACK_MODEL"),
    )


def load_model_config() -> ModelConfig:
    load_dotenv()
    try:
        max_tokens = int(os.environ.get("MICROCODE_MAX_TOKENS") or "8192")
    except ValueError:
        max_tokens = 8192
    return ModelConfig(
        api_key=_nonempty("MICROCODE_API_KEY"),
        base_url=(_nonempty("MICROCODE_BASE_URL") or "https://api.openai.com/v1").rstrip("/"),
        model=_nonempty("MICROCODE_MODEL") or "gpt-4o-mini",
        max_tokens=max(1, max_tokens),
        protocol=(os.environ.get("MICROCODE_API_PROTOCOL") or "auto").strip().lower(),
    )

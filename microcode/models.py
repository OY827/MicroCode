from __future__ import annotations

from microcode.anthropic_adapter import AnthropicModelAdapter
from microcode.config import ModelConfig, load_model_config, resolve_protocol
from microcode.mock_model import MockModelAdapter
from microcode.openai_adapter import OpenAIModelAdapter
from microcode.tooling import ToolRegistry
from microcode.types import ModelAdapter


def create_model_adapter(
    config: ModelConfig,
    tools: ToolRegistry,
    *,
    stream: bool = True,
) -> ModelAdapter:
    if not config.is_configured:
        return MockModelAdapter()
    if resolve_protocol(config) == "anthropic":
        return AnthropicModelAdapter(config, tools, stream=stream)
    return OpenAIModelAdapter(config, tools, stream=stream)


def model_label(model: ModelAdapter | None) -> str:
    if model is None:
        return ""
    if type(model).__name__ == "MockModelAdapter":
        return "mock"
    config = getattr(model, "config", None)
    if config is not None:
        return str(getattr(config, "model", "") or type(model).__name__)
    return type(model).__name__


def describe_model(model: ModelAdapter | None) -> str:
    return "\n".join(model_lines(model))


def model_lines(model: ModelAdapter | None) -> list[str]:
    if model is None:
        return ["model: (unknown)"]
    if type(model).__name__ == "MockModelAdapter":
        return ["model: offline mock (not a live network model)"]
    config = getattr(model, "config", None)
    if config is None:
        return [f"model: {type(model).__name__}"]
    protocol = resolve_protocol(config)
    return [
        f"model: {config.model} ({protocol})",
        f"address: {config.base_url}",
    ]


def apply_model_switch(
    current: ModelAdapter,
    tools: ToolRegistry,
    arg: str,
) -> tuple[ModelAdapter, str]:
    """Swap the adapter for this chat. Does not write .env."""

    name = arg.strip()
    stream = bool(getattr(current, "stream", True))
    if not name:
        return current, describe_model(current)
    lowered = name.lower()
    if lowered == "mock":
        return MockModelAdapter(), "Switched to offline mock. Next turn stays in this chat."
    if lowered == "live":
        config = load_model_config()
        if not config.is_configured:
            return current, "No API key in .env. Staying on the current model."
        adapter = create_model_adapter(config, tools, stream=stream)
        return adapter, f"Switched to {config.model} ({resolve_protocol(config)}). Next turn uses this model."

    base = getattr(current, "config", None)
    if base is None or not getattr(base, "is_configured", False):
        base = load_model_config()
    if not getattr(base, "is_configured", False):
        return current, "No API key. Cannot switch to a live model. Use /model mock."
    new_config = ModelConfig(
        api_key=base.api_key,
        base_url=base.base_url,
        model=name,
        max_tokens=base.max_tokens,
        protocol=base.protocol,
    )
    adapter = create_model_adapter(new_config, tools, stream=stream)
    return adapter, f"Switched to {name} ({resolve_protocol(new_config)}). Next turn uses this model."

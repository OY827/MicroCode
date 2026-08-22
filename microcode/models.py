from __future__ import annotations

from microcode.anthropic_adapter import AnthropicModelAdapter
from microcode.config import ModelConfig, resolve_protocol
from microcode.mock_model import MockModelAdapter
from microcode.openai_adapter import OpenAIModelAdapter
from microcode.tooling import ToolRegistry
from microcode.types import ModelAdapter


def create_model_adapter(config: ModelConfig, tools: ToolRegistry) -> ModelAdapter:
    if not config.is_configured:
        return MockModelAdapter()
    if resolve_protocol(config) == "anthropic":
        return AnthropicModelAdapter(config, tools)
    return OpenAIModelAdapter(config, tools)

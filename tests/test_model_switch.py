from microcode.config import ModelConfig
from microcode.mock_model import MockModelAdapter
from microcode.models import apply_model_switch, describe_model, model_label
from microcode.openai_adapter import OpenAIModelAdapter
from microcode.tooling import ToolRegistry


def test_describe_mock() -> None:
    assert model_label(MockModelAdapter()) == "mock"
    assert "offline mock" in describe_model(MockModelAdapter())


def test_switch_to_mock() -> None:
    current = OpenAIModelAdapter(
        ModelConfig(api_key="sk", base_url="https://api.openai.com/v1", model="gpt-4o-mini"),
        ToolRegistry([]),
        stream=False,
    )
    next_model, message = apply_model_switch(current, ToolRegistry([]), "mock")
    assert isinstance(next_model, MockModelAdapter)
    assert "offline mock" in message


def test_switch_name_reuses_current_endpoint() -> None:
    current = OpenAIModelAdapter(
        ModelConfig(api_key="sk", base_url="https://example.test/v1", model="gpt-4o-mini"),
        ToolRegistry([]),
        stream=False,
    )
    next_model, message = apply_model_switch(current, ToolRegistry([]), "gpt-4o")
    assert isinstance(next_model, OpenAIModelAdapter)
    assert next_model.config.model == "gpt-4o"
    assert next_model.config.base_url == "https://example.test/v1"
    assert "gpt-4o" in message


def test_switch_without_key_stays_put(monkeypatch) -> None:
    monkeypatch.setattr(
        "microcode.models.load_model_config",
        lambda: ModelConfig(api_key="", base_url="https://api.openai.com/v1", model="gpt-4o-mini"),
    )
    current = MockModelAdapter()
    next_model, message = apply_model_switch(current, ToolRegistry([]), "gpt-4o")
    assert next_model is current
    assert "No API key" in message

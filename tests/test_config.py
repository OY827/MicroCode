import os
from pathlib import Path

from microcode.config import load_dotenv, load_model_config


def test_load_dotenv_sets_missing_keys(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "MICROCODE_API_KEY=sk-test\nMICROCODE_MODEL=demo-model\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("MICROCODE_API_KEY", raising=False)
    monkeypatch.delenv("MICROCODE_MODEL", raising=False)

    loaded = load_dotenv(env_file)

    assert loaded == env_file
    assert os.environ["MICROCODE_API_KEY"] == "sk-test"
    assert os.environ["MICROCODE_MODEL"] == "demo-model"


def test_load_dotenv_does_not_override_existing_env(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("MICROCODE_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("MICROCODE_API_KEY", "from-process")

    load_dotenv(env_file)

    assert os.environ["MICROCODE_API_KEY"] == "from-process"


def test_load_dotenv_fills_blank_existing_keys(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("MICROCODE_API_KEY=sk-from-file\n", encoding="utf-8")
    monkeypatch.setenv("MICROCODE_API_KEY", "")

    load_dotenv(env_file)

    assert os.environ["MICROCODE_API_KEY"] == "sk-from-file"


def test_load_model_config_reads_microcode_vars(monkeypatch) -> None:
    monkeypatch.setenv("MICROCODE_API_KEY", "sk-live")
    monkeypatch.setenv("MICROCODE_BASE_URL", "https://api.deepseek.com/v1/")
    monkeypatch.setenv("MICROCODE_MODEL", "deepseek-chat")

    config = load_model_config()

    assert config.is_configured
    assert config.api_key == "sk-live"
    assert config.base_url == "https://api.deepseek.com/v1"
    assert config.model == "deepseek-chat"


def test_resolve_protocol_from_deepseek_anthropic_url() -> None:
    from microcode.config import ModelConfig, resolve_protocol

    config = ModelConfig(
        api_key="x",
        base_url="https://api.deepseek.com/anthropic",
        model="deepseek-v4-pro",
    )
    assert resolve_protocol(config) == "anthropic"

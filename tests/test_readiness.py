from microcode.config import FallbackConfig, ModelConfig
from microcode.main import main
from microcode.readiness import build_readiness_report


def _primary(**overrides) -> ModelConfig:
    data = {
        "api_key": "sk-live",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    }
    data.update(overrides)
    return ModelConfig(**data)


def test_ready_when_primary_and_fallback_keys_exist() -> None:
    report = build_readiness_report(
        config=_primary(),
        fallback=FallbackConfig(api_key="sk-spare"),
    )
    assert report.status == "ready"
    assert report.exit_code("blocked") == 0
    assert "https://api.openai.com/v1" in report.format()


def test_warning_without_any_key() -> None:
    report = build_readiness_report(
        config=_primary(api_key=""),
        fallback=FallbackConfig(),
    )
    assert report.status == "warning"
    assert "mock" in report.format().lower()
    assert report.exit_code("blocked") == 0
    assert report.exit_code("warning") == 2


def test_blocked_when_live_key_has_bad_url() -> None:
    report = build_readiness_report(
        config=_primary(base_url="not-a-url"),
        fallback=FallbackConfig(api_key="sk-spare"),
    )
    assert report.status == "blocked"
    assert report.exit_code() == 2
    assert "http" in report.format()


def test_warning_when_only_fallback_key() -> None:
    report = build_readiness_report(
        config=_primary(api_key=""),
        fallback=FallbackConfig(api_key="sk-spare", model="spare-model"),
    )
    assert report.status == "warning"
    assert any(check.name == "fallback key" and check.status == "pass" for check in report.checks)


def test_cli_readiness_json_exits_before_session(monkeypatch, capsys) -> None:
    monkeypatch.setattr("microcode.config.load_dotenv", lambda path=None: None)
    monkeypatch.setattr("microcode.main.load_dotenv", lambda: None)
    monkeypatch.delenv("MICROCODE_API_KEY", raising=False)
    monkeypatch.delenv("MICROCODE_FALLBACK_API_KEY", raising=False)
    code = main(["--readiness-json"])
    captured = capsys.readouterr()
    assert code == 0
    assert '"status": "warning"' in captured.out
    assert "session:" not in captured.err

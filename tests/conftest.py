import pytest


@pytest.fixture(autouse=True)
def _disable_dense_minilm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep pytest off the real MiniLM download/load path."""

    monkeypatch.setenv("MICROCODE_DENSE_MEMORY", "0")
    monkeypatch.setenv("MICROCODE_TURN_TAPE", "0")

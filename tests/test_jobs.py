import time
from pathlib import Path

from microcode.jobs import JobStore
from microcode.repl import run_repl
from microcode.tooling import ToolContext
from microcode.tools import create_default_tool_registry
from microcode.types import AgentStep, ChatMessage

SLEEP_WRITE = (
    'python -c "import time, pathlib; time.sleep(2); '
    "pathlib.Path('done.txt').write_text('ok', encoding='utf-8')\""
)


class ScriptedModel:
    def __init__(self, steps: list[AgentStep]) -> None:
        self._steps = steps
        self.calls = 0

    def next(self, messages: list[ChatMessage], on_text_delta=None) -> AgentStep:
        step = self._steps[self.calls]
        self.calls += 1
        return step


def _job_id(output: str) -> str:
    for line in output.splitlines():
        if line.startswith("started "):
            return line.split()[1]
    raise AssertionError(f"no job id in: {output}")


def test_background_returns_before_command_finishes(tmp_path: Path) -> None:
    tools = create_default_tool_registry()
    store = JobStore()
    context = ToolContext(cwd=str(tmp_path), jobs=store)
    started = time.perf_counter()
    result = tools.execute(
        "run_command",
        {"command": SLEEP_WRITE, "background": True},
        context,
    )
    elapsed = time.perf_counter() - started
    try:
        assert result.ok
        assert elapsed < 1.5
        job_id = _job_id(result.output)
        assert job_id == "job_1"
        assert not (tmp_path / "done.txt").exists()
        listed = tools.execute("list_files", {}, context)
        assert listed.ok
        collected = tools.execute("await_job", {"job_id": job_id, "timeout": 10}, context)
        assert collected.ok
        assert (tmp_path / "done.txt").read_text(encoding="utf-8") == "ok"
        assert "exit_code: 0" in collected.output
        again = tools.execute("await_job", {"job_id": "job_1"}, context)
        assert again.ok
        assert "exit_code: 0" in again.output
    finally:
        store.kill_all()


def test_await_job_timeout_leaves_process_running(tmp_path: Path) -> None:
    tools = create_default_tool_registry()
    store = JobStore()
    context = ToolContext(cwd=str(tmp_path), jobs=store)
    tools.execute(
        "run_command",
        {
            "command": 'python -c "import time; time.sleep(30)"',
            "background": True,
        },
        context,
    )
    try:
        result = tools.execute("await_job", {"job_id": "job_1", "timeout": 1}, context)
        assert not result.ok
        assert "still running" in result.output
        assert "job_1  running" in store.format()
    finally:
        report = store.kill_all()
        assert "job_1" in report
        assert "running" not in store.format()


def test_unknown_and_rejected_background(tmp_path: Path) -> None:
    tools = create_default_tool_registry()
    store = JobStore()
    missing = tools.execute(
        "await_job",
        {"job_id": "job_9"},
        ToolContext(cwd=str(tmp_path), jobs=store),
    )
    assert not missing.ok
    assert "Unknown job" in missing.output

    rejected = tools.execute(
        "run_command",
        {"command": "python -c pass", "background": True},
        ToolContext(cwd=str(tmp_path), jobs=store, on_approve=lambda _summary: False),
    )
    assert not rejected.ok
    assert "rejected" in rejected.output
    assert store.format() == "No jobs."


def test_repl_jobs_command(tmp_path: Path, capsys) -> None:
    store = JobStore()
    lines = iter(["/jobs", "/exit"])
    run_repl(
        model=ScriptedModel([]),
        tools=create_default_tool_registry(),
        messages=[{"role": "system", "content": "sys"}],
        cwd=str(tmp_path),
        jobs=store,
        read_line=lambda: next(lines) + "\n",
        on_assistant_message=lambda _text: None,
    )
    err = capsys.readouterr().err
    assert "No jobs." in err

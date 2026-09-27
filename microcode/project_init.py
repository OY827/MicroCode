from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from microcode.memory import ProjectMemory

INIT_RELATIVE = Path(".microcode") / "INIT.md"
INIT_TAG = "project-init"
INIT_MAX_CHARS = 4_000

ALWAYS_LEAVE_ALONE = (".env", ".git")
MAYBE_LEAVE_ALONE = (".venv", "node_modules", "__pycache__", ".microcode", "dist", "build")


@dataclass
class InitReport:
    how_to_test: list[str] = field(default_factory=list)
    leave_alone: list[str] = field(default_factory=list)
    layout: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)

    def markdown(self) -> str:
        lines = [
            "# Project conventions",
            "",
            "Written by `/init`. Safe to edit. Running `/init` again overwrites this file.",
            "",
            "## How to test",
        ]
        if self.how_to_test:
            lines.extend(f"- {item}" for item in self.how_to_test)
        else:
            lines.append("- No test runner detected. Ask before inventing one.")
        lines.extend(["", "## Do not touch"])
        lines.extend(f"- {item}" for item in self.leave_alone)
        if self.layout:
            lines.extend(["", "## Top level"])
            lines.extend(f"- {item}" for item in self.layout)
        if self.hints:
            lines.extend(["", "## Notes"])
            lines.extend(f"- {item}" for item in self.hints)
        return "\n".join(lines).rstrip() + "\n"

    def memory_line(self) -> str:
        tests = self.how_to_test[0] if self.how_to_test else "No test command detected."
        leave = ", ".join(self.leave_alone[:6])
        return f"Project init: {tests} Do not touch {leave}."


def _read(path: Path, limit: int = 8_000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:limit]
    except OSError:
        return ""


def scan_project(root: str | Path) -> InitReport:
    workspace = Path(root)
    report = InitReport()
    report.leave_alone = list(ALWAYS_LEAVE_ALONE)
    for name in MAYBE_LEAVE_ALONE:
        if (workspace / name).exists() and name not in report.leave_alone:
            report.leave_alone.append(name)

    pyproject = _read(workspace / "pyproject.toml")
    package_json = workspace / "package.json"
    tests_dir = workspace / "tests"
    if (workspace / "pytest.ini").exists() or (workspace / "conftest.py").exists() or tests_dir.is_dir():
        report.how_to_test.append("Python tests: use test_runner or `pytest`.")
    if pyproject:
        report.hints.append("Python project (pyproject.toml).")
        if "[tool.pytest" in pyproject or "pytest" in pyproject:
            if not any("pytest" in item for item in report.how_to_test):
                report.how_to_test.append("Python tests: pytest is configured in pyproject.toml.")
    if (workspace / "requirements.txt").exists() and not pyproject:
        report.hints.append("Python project (requirements.txt).")
    if package_json.is_file():
        report.hints.append("Node project (package.json).")
        try:
            data = json.loads(_read(package_json, 20_000) or "{}")
        except json.JSONDecodeError:
            data = {}
        scripts = data.get("scripts") if isinstance(data, dict) else None
        if isinstance(scripts, dict) and scripts.get("test"):
            report.how_to_test.append(f"Node tests: npm test ({scripts['test']}).")
    if (workspace / "go.mod").exists():
        report.how_to_test.append("Go tests: `go test ./...`.")
        report.hints.append("Go project (go.mod).")
    if (workspace / "Cargo.toml").exists():
        report.how_to_test.append("Rust tests: `cargo test`.")
        report.hints.append("Rust project (Cargo.toml).")
    makefile = _read(workspace / "Makefile")
    if makefile and re.search(r"(?m)^test:", makefile):
        report.how_to_test.append("Makefile has a `test` target: `make test`.")

    try:
        entries = sorted(workspace.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
    except OSError:
        entries = []
    for entry in entries[:12]:
        if entry.name.startswith(".") and entry.name not in {".microcode"}:
            continue
        kind = "dir" if entry.is_dir() else "file"
        report.layout.append(f"{kind} {entry.name}")
    return report


def write_init_file(root: str | Path, report: InitReport | None = None) -> Path:
    workspace = Path(root)
    scanned = report or scan_project(workspace)
    target = workspace / INIT_RELATIVE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(scanned.markdown(), encoding="utf-8")
    return target


def load_init_markdown(root: str | Path, *, max_chars: int = INIT_MAX_CHARS) -> str:
    path = Path(root) / INIT_RELATIVE
    if not path.is_file():
        return ""
    text = _read(path, max_chars + 1).strip()
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "\n... (init truncated)"
    return text


def run_init(root: str | Path, memory: ProjectMemory | None = None) -> str:
    """Scan the workspace, write INIT.md, and remember a one-line note."""

    report = scan_project(root)
    path = write_init_file(root, report)
    if memory is not None:
        memory.manager.upsert_tagged(
            report.memory_line(),
            tag=INIT_TAG,
            extra_tags=["init"],
        )
    return f"Wrote {path}\n\n{report.markdown()}"

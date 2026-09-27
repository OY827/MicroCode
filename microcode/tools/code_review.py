from __future__ import annotations

import ast
from dataclasses import dataclass

from microcode.tooling import ToolDefinition, ToolResult
from microcode.tools.code_nav import iter_python_files
from microcode.workspace import relative_workspace_path, resolve_tool_path

CHECK_KINDS = ("all", "imports", "excepts", "complexity")
LONG_FUNCTION_LINES = 50
MAX_ISSUES = 80


@dataclass(frozen=True)
class ReviewIssue:
    severity: str
    kind: str
    line: int
    message: str


def _bound_import_names(tree: ast.AST) -> list[tuple[str, int]]:
    names: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".", 1)[0]
                names.append((bound, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    continue
                bound = alias.asname or alias.name
                names.append((bound, node.lineno))
    return names


def _used_identifiers(tree: ast.AST) -> set[str]:
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


def check_unused_imports(tree: ast.AST) -> list[ReviewIssue]:
    used = _used_identifiers(tree)
    issues: list[ReviewIssue] = []
    seen: set[str] = set()
    for name, lineno in _bound_import_names(tree):
        if name in seen:
            continue
        seen.add(name)
        if name not in used:
            issues.append(
                ReviewIssue("warning", "unused_import", lineno, f"imported name '{name}' is never used")
            )
    return issues


def check_excepts(tree: ast.AST) -> list[ReviewIssue]:
    issues: list[ReviewIssue] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if node.type is None:
            issues.append(
                ReviewIssue("warning", "bare_except", node.lineno, "bare except swallows all exceptions")
            )
        body = node.body
        if len(body) == 1 and isinstance(body[0], ast.Pass):
            issues.append(
                ReviewIssue("warning", "empty_except", node.lineno, "except body is only 'pass'")
            )
    return issues


def check_long_functions(tree: ast.AST) -> list[ReviewIssue]:
    issues: list[ReviewIssue] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        end = getattr(node, "end_lineno", None)
        if end is None:
            continue
        length = end - node.lineno + 1
        if length > LONG_FUNCTION_LINES:
            issues.append(
                ReviewIssue(
                    "warning",
                    "long_function",
                    node.lineno,
                    f"function '{node.name}' is {length} lines (limit {LONG_FUNCTION_LINES})",
                )
            )
    return issues


def review_source(source: str, *, filename: str = "<unknown>") -> list[ReviewIssue]:
    tree = ast.parse(source, filename=filename)
    issues = check_unused_imports(tree) + check_excepts(tree) + check_long_functions(tree)
    issues.sort(key=lambda item: (item.line, item.kind))
    return issues


def _validate(input_data: dict) -> dict:
    if "path" in input_data and not isinstance(input_data["path"], str):
        raise ValueError("path must be a string")
    checks = input_data.get("checks", "all")
    if checks not in CHECK_KINDS:
        raise ValueError(f"checks must be one of: {', '.join(CHECK_KINDS)}")
    return {"path": input_data.get("path", "."), "checks": checks}


def _run(input_data: dict, context) -> ToolResult:
    target = resolve_tool_path(context, input_data["path"])
    if not target.exists():
        return ToolResult(ok=False, output=f"Path does not exist: {input_data['path']}")
    if target.is_file() and target.suffix != ".py":
        return ToolResult(ok=False, output=f"Not a Python file: {input_data['path']}")

    checks = input_data["checks"]
    rows: list[str] = []
    files_reviewed = 0
    for py_file in iter_python_files(target):
        try:
            source = py_file.read_text(encoding="utf-8")
            issues = review_source(source, filename=str(py_file))
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue
        files_reviewed += 1
        if checks == "imports":
            issues = [item for item in issues if item.kind == "unused_import"]
        elif checks == "excepts":
            issues = [item for item in issues if item.kind in {"bare_except", "empty_except"}]
        elif checks == "complexity":
            issues = [item for item in issues if item.kind == "long_function"]
        try:
            rel = relative_workspace_path(context.cwd, py_file)
        except ValueError:
            rel = py_file.name
        for issue in issues:
            rows.append(f"{issue.severity}  {rel}:{issue.line}  {issue.kind}  {issue.message}")
            if len(rows) >= MAX_ISSUES:
                break
        if len(rows) >= MAX_ISSUES:
            break

    header = [
        f"path: {input_data['path']}",
        f"files: {files_reviewed}",
        f"issues: {len(rows)}",
    ]
    if not rows:
        return ToolResult(ok=True, output="\n".join(header) + "\nNo issues found.")
    return ToolResult(ok=True, output="\n".join(header + rows))


code_review_tool = ToolDefinition(
    name="code_review",
    description=(
        "Static Python review: unused imports, bare/empty except, and functions longer than "
        f"{LONG_FUNCTION_LINES} lines. Does not run tests. Use test_runner to verify behavior."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "checks": {"type": "string", "enum": list(CHECK_KINDS)},
        },
    },
    validator=_validate,
    run=_run,
)

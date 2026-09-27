from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from pathlib import Path

from microcode.tooling import ToolDefinition, ToolResult
from microcode.tools.list_files import SKIP_NAMES
from microcode.workspace import relative_workspace_path, resolve_tool_path

SKIP_DIRS = SKIP_NAMES | {"node_modules", "dist", "build", ".tox", "venv"}
SYMBOL_TYPES = ("all", "function", "class")
MAX_FILES = 500
MAX_HITS = 200


@dataclass(frozen=True)
class SymbolHit:
    kind: str
    name: str
    file: str
    line: int
    extra: str = ""


def iter_python_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix == ".py" else []
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS]
        for name in filenames:
            if name.endswith(".py") and name not in {".env"}:
                found.append(Path(dirpath) / name)
            if len(found) >= MAX_FILES:
                return sorted(found)
    return sorted(found)


def _function_args(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    names = [arg.arg for arg in node.args.args]
    return "(" + ", ".join(names) + ")"


def extract_symbols(source: str, *, filename: str = "<unknown>") -> list[tuple[str, str, int, str]]:
    """Return (kind, name, line, extra) for function and class definitions."""

    tree = ast.parse(source, filename=filename)
    hits: list[tuple[str, str, int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            hits.append(("function", node.name, node.lineno, _function_args(node)))
        elif isinstance(node, ast.ClassDef):
            bases = ", ".join(ast.unparse(base) for base in node.bases)
            extra = f"({bases})" if bases else ""
            hits.append(("class", node.name, node.lineno, extra))
    return hits


def extract_name_uses(source: str, symbol: str, *, filename: str = "<unknown>") -> list[tuple[str, int]]:
    """Return (kind, line) for definitions and identifier uses of symbol. Strings/comments are ignored."""

    tree = ast.parse(source, filename=filename)
    hits: list[tuple[str, int]] = []

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if node.name == symbol:
                hits.append(("definition", node.lineno))
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            if node.name == symbol:
                hits.append(("definition", node.lineno))
            self.generic_visit(node)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            if node.name == symbol:
                hits.append(("definition", node.lineno))
            self.generic_visit(node)

        def visit_Name(self, node: ast.Name) -> None:
            if node.id == symbol:
                hits.append(("use", node.lineno))

        def visit_Attribute(self, node: ast.Attribute) -> None:
            if node.attr == symbol:
                hits.append(("use", node.lineno))
            self.generic_visit(node)

    Visitor().visit(tree)
    return hits


def _line_preview(lines: list[str], lineno: int) -> str:
    if lineno < 1 or lineno > len(lines):
        return ""
    return lines[lineno - 1].strip()


def _rel(cwd: str, path: Path) -> str:
    try:
        return relative_workspace_path(cwd, path)
    except ValueError:
        return path.name


def _run_find_symbols(input_data: dict, context) -> ToolResult:
    target = resolve_tool_path(context, input_data["path"])
    if not target.exists():
        return ToolResult(ok=False, output=f"Path does not exist: {input_data['path']}")
    wanted_type = input_data["symbol_type"]
    wanted_name = input_data["name"]
    rows: list[str] = []
    for py_file in iter_python_files(target):
        try:
            source = py_file.read_text(encoding="utf-8")
            symbols = extract_symbols(source, filename=str(py_file))
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue
        rel = _rel(context.cwd, py_file)
        for kind, name, line, extra in symbols:
            if wanted_type != "all" and kind != wanted_type:
                continue
            if wanted_name and name != wanted_name:
                continue
            rows.append(f"{rel}:{line}  {kind}  {name}{extra}")
            if len(rows) >= MAX_HITS:
                break
        if len(rows) >= MAX_HITS:
            break
    if not rows:
        label = wanted_name or "symbols"
        return ToolResult(ok=True, output=f"No {label} found in {input_data['path']}")
    header = f"{len(rows)} symbol(s) in {input_data['path']}"
    return ToolResult(ok=True, output=header + "\n" + "\n".join(rows))


def _run_find_references(input_data: dict, context) -> ToolResult:
    target = resolve_tool_path(context, input_data["path"])
    if not target.exists():
        return ToolResult(ok=False, output=f"Path does not exist: {input_data['path']}")
    symbol = input_data["name"]
    rows: list[str] = []
    for py_file in iter_python_files(target):
        try:
            source = py_file.read_text(encoding="utf-8")
            lines = source.splitlines()
            uses = extract_name_uses(source, symbol, filename=str(py_file))
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue
        rel = _rel(context.cwd, py_file)
        for kind, line in uses:
            preview = _line_preview(lines, line)
            rows.append(f"{rel}:{line}  {kind}  {preview}")
            if len(rows) >= MAX_HITS:
                break
        if len(rows) >= MAX_HITS:
            break
    if not rows:
        return ToolResult(ok=True, output=f"No references to {symbol} in {input_data['path']}")
    header = f"{len(rows)} reference(s) to {symbol}"
    return ToolResult(ok=True, output=header + "\n" + "\n".join(rows))


def _validate_find_symbols(input_data: dict) -> dict:
    if "path" in input_data and not isinstance(input_data["path"], str):
        raise ValueError("path must be a string")
    symbol_type = input_data.get("symbol_type", "all")
    if symbol_type not in SYMBOL_TYPES:
        raise ValueError(f"symbol_type must be one of: {', '.join(SYMBOL_TYPES)}")
    name = input_data.get("name")
    if name is not None and not isinstance(name, str):
        raise ValueError("name must be a string")
    return {
        "path": input_data.get("path", "."),
        "symbol_type": symbol_type,
        "name": (name or "").strip(),
    }


def _validate_find_references(input_data: dict) -> dict:
    name = input_data.get("name") or input_data.get("symbol_name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name is required")
    if "path" in input_data and not isinstance(input_data["path"], str):
        raise ValueError("path must be a string")
    return {"name": name.strip(), "path": input_data.get("path", ".")}


find_symbols_tool = ToolDefinition(
    name="find_symbols",
    description=(
        "Find Python function and class definitions via AST (not text search). "
        "Optional name filters to one identifier. Comments and strings are ignored."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "name": {"type": "string", "description": "Only this identifier, e.g. login."},
            "symbol_type": {"type": "string", "enum": list(SYMBOL_TYPES)},
        },
    },
    validator=_validate_find_symbols,
    run=_run_find_symbols,
)

find_references_tool = ToolDefinition(
    name="find_references",
    description=(
        "Find Python definitions and identifier uses of a name via AST. "
        "Does not match the name inside comments or string literals. "
        "Use find_symbols first to locate a definition."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "symbol_name": {"type": "string"},
            "path": {"type": "string"},
        },
        "required": ["name"],
    },
    validator=_validate_find_references,
    run=_run_find_references,
)

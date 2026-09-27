from __future__ import annotations

from difflib import unified_diff
from pathlib import Path

from microcode.tooling import ToolContext, ToolResult
from microcode.workspace import relative_workspace_path, resolve_tool_path

PROTECTED_NAMES = {".env"}
PREVIEW_LINE_LIMIT = 80


def apply_file_change(
    context: ToolContext,
    input_path: str,
    new_content: str,
) -> ToolResult:
    target = resolve_tool_path(context, input_path)
    if target.name in PROTECTED_NAMES:
        return ToolResult(ok=False, output=f"Refusing to write protected file: {input_path}")

    old_content = ""
    existed = target.exists()
    if existed:
        if target.is_dir():
            return ToolResult(ok=False, output=f"Path is a directory: {input_path}")
        try:
            old_content = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ToolResult(ok=False, output=f"File {input_path} appears to be binary.")
        except OSError as error:
            return ToolResult(ok=False, output=str(error))

    if old_content == new_content:
        return ToolResult(ok=True, output=f"No changes for {input_path}")

    preview = render_unified_diff(input_path, old_content, new_content)
    if context.on_write_preview:
        context.on_write_preview(input_path, preview)
    relpath = relative_workspace_path(context.cwd, target)
    revised = context.revise_write(input_path, new_content, preview, kind="write", key=relpath)
    if revised is None:
        return ToolResult(ok=False, output=context.reject_text(f"write to {input_path}"))
    edited = revised != new_content
    new_content = revised
    if edited:
        preview = render_unified_diff(input_path, old_content, new_content)

    if context.checkpoint is not None:
        context.checkpoint.record_target(
            context.cwd,
            target,
            existed=existed,
            content=old_content,
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(new_content, encoding="utf-8")
    action = "Updated" if existed else "Created"
    note = " edited before write" if edited else ""
    return ToolResult(
        ok=True,
        output=f"{action} {input_path} ({len(new_content)} chars){note}\n\n{preview}",
    )


def render_unified_diff(path: str, old_content: str, new_content: str) -> str:
    diff = list(
        unified_diff(
            old_content.splitlines(),
            new_content.splitlines(),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="",
        )
    )
    if not diff:
        return "(no textual diff)"
    if len(diff) > PREVIEW_LINE_LIMIT:
        omitted = len(diff) - PREVIEW_LINE_LIMIT
        diff = diff[:PREVIEW_LINE_LIMIT] + [f"... ({omitted} more diff lines omitted)"]
    return "\n".join(diff)


def load_text_file(target: Path, input_path: str) -> str:
    if not target.exists():
        raise FileNotFoundError(f"File does not exist: {input_path}")
    if target.is_dir():
        raise IsADirectoryError(f"Path is a directory: {input_path}")
    return target.read_text(encoding="utf-8")

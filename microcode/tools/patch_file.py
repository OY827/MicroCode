from __future__ import annotations

from microcode.file_edit import apply_file_change, load_text_file
from microcode.tooling import ToolDefinition, ToolResult
from microcode.workspace import resolve_tool_path


def _validate(input_data: dict) -> dict:
    path = input_data.get("path")
    replacements = input_data.get("replacements")
    if not isinstance(path, str) or not path:
        raise ValueError("path is required")
    if not isinstance(replacements, list) or not replacements:
        raise ValueError("replacements must be a non-empty list")
    normalized = []
    for replacement in replacements:
        if not isinstance(replacement, dict):
            raise ValueError("replacement entries must be objects")
        search = replacement.get("search")
        replace = replacement.get("replace")
        replace_all = bool(replacement.get("replace_all", replacement.get("replaceAll", False)))
        if not isinstance(search, str) or not search:
            raise ValueError("replacement search must be a non-empty string")
        if not isinstance(replace, str):
            raise ValueError("replacement replace must be a string")
        normalized.append(
            {
                "search": search.replace("\r\n", "\n"),
                "replace": replace.replace("\r\n", "\n"),
                "replace_all": replace_all,
            }
        )
    return {"path": path, "replacements": normalized}


def _apply_replacements(content: str, replacements: list[dict], path: str) -> tuple[str, list[str]] | ToolResult:
    """Apply every hunk in memory. Fail the whole patch if any hunk is missing or ambiguous."""

    applied: list[str] = []
    current = content
    for index, replacement in enumerate(replacements, start=1):
        search = replacement["search"]
        count = current.count(search)
        if count == 0:
            return ToolResult(ok=False, output=f"Replacement {index} not found in {path}")
        if count > 1 and not replacement["replace_all"]:
            return ToolResult(
                ok=False,
                output=(
                    f"Replacement {index} occurs {count} times in {path}. "
                    "Pass replace_all=true or use a more specific search string."
                ),
            )
        if replacement["replace_all"]:
            current = current.replace(search, replacement["replace"])
            applied.append(f"#{index} replace_all")
        else:
            current = current.replace(search, replacement["replace"], 1)
            applied.append(f"#{index} replace")
    return current, applied


def _run(input_data: dict, context) -> ToolResult:
    target = resolve_tool_path(context, input_data["path"])
    try:
        content = load_text_file(target, input_data["path"])
    except (OSError, UnicodeDecodeError) as error:
        return ToolResult(ok=False, output=str(error))

    patched = _apply_replacements(content, input_data["replacements"], input_data["path"])
    if isinstance(patched, ToolResult):
        return patched
    next_content, applied = patched
    result = apply_file_change(context, input_data["path"], next_content)
    if not result.ok:
        return result
    return ToolResult(
        ok=True,
        output=f"Patched {input_data['path']} with {len(applied)} replacement(s): {', '.join(applied)}\n\n{result.output}",
    )


patch_file_tool = ToolDefinition(
    name="patch_file",
    description=(
        "Apply multiple exact substring replacements to one UTF-8 file in a single write. "
        "Each search must match once unless that hunk sets replace_all. "
        "If any hunk fails, the file is left unchanged."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "replacements": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "search": {"type": "string"},
                        "replace": {"type": "string"},
                        "replace_all": {"type": "boolean"},
                        "replaceAll": {"type": "boolean"},
                    },
                    "required": ["search", "replace"],
                },
            },
        },
        "required": ["path", "replacements"],
    },
    validator=_validate,
    run=_run,
)

from __future__ import annotations

from microcode.file_edit import apply_file_change, load_text_file
from microcode.tooling import ToolDefinition, ToolResult
from microcode.workspace import resolve_tool_path


def _validate(input_data: dict) -> dict:
    path = input_data.get("path")
    search = input_data.get("search", input_data.get("old"))
    replace = input_data.get("replace", input_data.get("new"))
    replace_all = bool(input_data.get("replace_all", input_data.get("replaceAll", False)))
    if not isinstance(path, str) or not path:
        raise ValueError("path is required")
    if not isinstance(search, str) or not search:
        raise ValueError("search must be a non-empty string")
    if not isinstance(replace, str):
        raise ValueError("replace must be a string")
    return {
        "path": path,
        "search": search.replace("\r\n", "\n"),
        "replace": replace.replace("\r\n", "\n"),
        "replace_all": replace_all,
    }


def _run(input_data: dict, context) -> ToolResult:
    target = resolve_tool_path(context, input_data["path"])
    try:
        content = load_text_file(target, input_data["path"])
    except (OSError, UnicodeDecodeError) as error:
        return ToolResult(ok=False, output=str(error))

    search = input_data["search"]
    count = content.count(search)
    if count == 0:
        return ToolResult(ok=False, output=f"Text not found in {input_data['path']}")
    if count > 1 and not input_data["replace_all"]:
        return ToolResult(
            ok=False,
            output=(
                f"Search text occurs {count} times in {input_data['path']}. "
                "Pass replace_all=true or use a more specific search string."
            ),
        )

    if input_data["replace_all"]:
        next_content = content.replace(search, input_data["replace"])
    else:
        next_content = content.replace(search, input_data["replace"], 1)
    return apply_file_change(context, input_data["path"], next_content)


edit_file_tool = ToolDefinition(
    name="edit_file",
    description=(
        "Replace a unique substring in a UTF-8 text file. "
        "Fails if the search text is missing or occurs more than once unless replace_all is true."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "search": {"type": "string"},
            "replace": {"type": "string"},
            "old": {"type": "string"},
            "new": {"type": "string"},
            "replace_all": {"type": "boolean"},
        },
        "required": ["path"],
    },
    validator=_validate,
    run=_run,
)

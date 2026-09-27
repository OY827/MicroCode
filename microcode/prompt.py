from __future__ import annotations

from microcode.compact import COMPACT_HINT
from microcode.memory import ProjectMemory
from microcode.project_init import load_init_markdown
from microcode.skills import discover_skills, format_skill_catalog


def build_system_prompt(
    base: str,
    cwd: str,
    *,
    query: str | None = None,
    memory: ProjectMemory | None = None,
) -> str:
    """Compose the system message: base person + memory + /init notes + skill catalog."""

    store = memory if memory is not None else ProjectMemory(cwd)
    text = store.inject(base, query=query)
    init = load_init_markdown(cwd)
    if init:
        text = f"{text.rstrip()}\n\n# Workspace init\n{init}"
    catalog = format_skill_catalog(discover_skills(cwd))
    if catalog:
        text = f"{text.rstrip()}\n\n{catalog}"
    return f"{text.rstrip()}\n\n{COMPACT_HINT}"

from __future__ import annotations

from microcode.memory import ProjectMemory
from microcode.skills import discover_skills, format_skill_catalog


def build_system_prompt(base: str, cwd: str) -> str:
    """Compose the system message: base person + memory + skill catalog."""

    text = ProjectMemory(cwd).inject(base)
    catalog = format_skill_catalog(discover_skills(cwd))
    if catalog:
        text = f"{text.rstrip()}\n\n{catalog}"
    return text

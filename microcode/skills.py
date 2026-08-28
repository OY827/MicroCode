from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MAX_SKILL_CHARS = 32_000
SKILLS_DIRNAME = "skills"


@dataclass(frozen=True, slots=True)
class SkillSummary:
    name: str
    description: str
    path: str


@dataclass(frozen=True, slots=True)
class LoadedSkill(SkillSummary):
    content: str


def extract_description(markdown: str) -> str:
    normalized = markdown.replace("\r\n", "\n")
    paragraphs = [block.strip() for block in normalized.split("\n\n") if block.strip()]
    for block in paragraphs:
        if block.startswith("#"):
            continue
        for line in [part.strip() for part in block.split("\n")]:
            if line and not line.startswith("#"):
                return line.replace("`", "")
    return "No description provided."


def skills_root(cwd: str | Path) -> Path:
    return Path(cwd) / ".microcode" / SKILLS_DIRNAME


def discover_skills(cwd: str | Path) -> list[SkillSummary]:
    root = skills_root(cwd)
    if not root.is_dir():
        return []
    found: list[SkillSummary] = []
    for entry in sorted(root.iterdir(), key=lambda item: item.name.lower()):
        if not entry.is_dir():
            continue
        skill = load_skill(cwd, entry.name)
        if skill is not None:
            found.append(SkillSummary(name=skill.name, description=skill.description, path=skill.path))
    return found


def load_skill(cwd: str | Path, name: str) -> LoadedSkill | None:
    cleaned = _safe_skill_name(name)
    if cleaned is None:
        return None
    root = skills_root(cwd).resolve()
    target = (root / cleaned / "SKILL.md").resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return None
    if not target.is_file():
        return None
    try:
        content = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    if len(content) > MAX_SKILL_CHARS:
        content = content[:MAX_SKILL_CHARS].rstrip() + "\n... (skill truncated)"
    return LoadedSkill(
        name=cleaned,
        description=extract_description(content),
        path=str(target),
        content=content,
    )


def format_skill_catalog(skills: list[SkillSummary]) -> str:
    if not skills:
        return ""
    lines = [
        "## Local skills",
        "When a task matches a skill below, call load_skill with that name before following it.",
        "Available skills:",
    ]
    for skill in skills:
        lines.append(f"- {skill.name}: {skill.description}")
    return "\n".join(lines)


def format_skill_list(skills: list[SkillSummary]) -> str:
    if not skills:
        return (
            "No local skills yet.\n"
            "Add .microcode/skills/<name>/SKILL.md"
        )
    lines = [f"{skill.name}  {skill.description}" for skill in skills]
    return "\n".join(lines)


def _safe_skill_name(name: str) -> str | None:
    cleaned = name.strip()
    if not cleaned or cleaned in {".", ".."}:
        return None
    if any(sep in cleaned for sep in ("/", "\\", ":")):
        return None
    return cleaned

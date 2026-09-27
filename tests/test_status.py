from pathlib import Path

from microcode.memory import ProjectMemory
from microcode.mock_model import MockModelAdapter
from microcode.permissions import PermissionStore
from microcode.session import SessionStore
from microcode.status import format_status
from microcode.types import AgentStep
from microcode.usage import UsageLedger


def test_status_lists_live_chat_facts(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    session = store.create(str(tmp_path), [{"role": "system", "content": "sys"}])
    memory = ProjectMemory(tmp_path)
    memory.append("Use pytest")
    permissions = PermissionStore(tmp_path / "perm.json")
    permissions.allow("write", "notes.txt")
    text = format_status(
        cwd=str(tmp_path),
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ],
        session=session,
        memory=memory,
        permissions=permissions,
        model=MockModelAdapter(),
    )
    assert f"workspace: {tmp_path}" in text
    assert f"session: {session.id}" in text
    assert "offline mock" in text
    assert "you spoke: 1 time(s)" in text
    assert "memory notes: 1" in text
    assert "1 write path(s)" in text
    assert "permission mode: ask" in text
    assert "can redo: 0 step(s)" in text
    assert "usage:" not in text
    usage = UsageLedger()
    usage.record(
        model="mock",
        usage=None,
        messages=[{"role": "user", "content": "hello"}],
        step=AgentStep(type="assistant", content="ok"),
    )
    with_usage = format_status(
        cwd=str(tmp_path),
        messages=[{"role": "user", "content": "hello"}],
        session=session,
        memory=memory,
        permissions=permissions,
        model=MockModelAdapter(),
        usage=usage,
    )
    assert "usage:" in with_usage

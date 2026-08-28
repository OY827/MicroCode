from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

MAX_OUTPUT_CHARS = 8000


def format_command_output(stdout: str, stderr: str, returncode: int) -> str:
    parts = [f"exit_code: {returncode}"]
    if stdout:
        parts.extend(["", "stdout:", stdout])
    if stderr:
        parts.extend(["", "stderr:", stderr])
    text = "\n".join(parts).strip()
    if len(text) > MAX_OUTPUT_CHARS:
        omitted = len(text) - MAX_OUTPUT_CHARS
        text = text[:MAX_OUTPUT_CHARS] + f"\n... ({omitted} more chars omitted)"
    return text


@dataclass
class Job:
    id: str
    command: str
    proc: subprocess.Popen | None
    stdout_path: Path
    stderr_path: Path
    stdout_handle: object | None
    stderr_handle: object | None
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""

    def running(self) -> bool:
        self.refresh()
        return self.returncode is None

    def refresh(self) -> None:
        if self.returncode is not None or self.proc is None:
            return
        if self.proc.poll() is not None:
            self._harvest()

    def collect(self, timeout: int) -> tuple[bool, str]:
        """Wait up to timeout seconds. Returns (finished, text)."""

        self.refresh()
        if self.returncode is not None:
            return True, format_command_output(self.stdout, self.stderr, self.returncode)
        if self.proc is None:
            return True, f"Job {self.id} has no process."
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, f"still running: {self.id}\n{self.command}"
        self._harvest()
        assert self.returncode is not None
        return True, format_command_output(self.stdout, self.stderr, self.returncode)

    def kill(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            _stop_process(self.proc)
        if self.returncode is None and self.proc is not None:
            self._harvest()
        else:
            self._close_handles()
            self._unlink_temps()

    def _harvest(self) -> None:
        if self.proc is None:
            return
        code = self.proc.poll()
        if code is None:
            return
        self._close_handles()
        self.stdout = _read_text(self.stdout_path)
        self.stderr = _read_text(self.stderr_path)
        self.returncode = code
        self._unlink_temps()
        self.proc = None

    def _close_handles(self) -> None:
        for handle in (self.stdout_handle, self.stderr_handle):
            if handle is None:
                continue
            try:
                handle.close()
            except OSError:
                pass
        self.stdout_handle = None
        self.stderr_handle = None

    def _unlink_temps(self) -> None:
        for path in (self.stdout_path, self.stderr_path):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


@dataclass
class JobStore:
    """In-memory background shell jobs for this process."""

    _jobs: dict[str, Job] = field(default_factory=dict)
    _next_id: int = 1

    def start(self, command: str, cwd: str) -> Job:
        out_fd, out_name = tempfile.mkstemp(prefix="microcode-job-out-")
        err_fd, err_name = tempfile.mkstemp(prefix="microcode-job-err-")
        out_handle = os.fdopen(out_fd, "wb")
        err_handle = os.fdopen(err_fd, "wb")
        proc = subprocess.Popen(
            command,
            shell=True,
            cwd=cwd,
            stdout=out_handle,
            stderr=err_handle,
        )
        job_id = f"job_{self._next_id}"
        self._next_id += 1
        job = Job(
            id=job_id,
            command=command,
            proc=proc,
            stdout_path=Path(out_name),
            stderr_path=Path(err_name),
            stdout_handle=out_handle,
            stderr_handle=err_handle,
        )
        self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id.strip())

    def format(self) -> str:
        if not self._jobs:
            return "No jobs."
        lines = ["jobs:"]
        for job in self._jobs.values():
            job.refresh()
            if job.returncode is None:
                status = "running"
            else:
                status = f"exit {job.returncode}"
            lines.append(f"  {job.id}  {status}  {job.command}")
        return "\n".join(lines)

    def kill_all(self) -> str:
        killed = []
        for job in self._jobs.values():
            if job.running():
                job.kill()
                killed.append(job.id)
            else:
                job.refresh()
        if not killed:
            return "No running jobs."
        return "Killed " + ", ".join(killed)


def jobs_from(context) -> JobStore:
    if context.jobs is None:
        context.jobs = JobStore()
    return context.jobs


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""


def _stop_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            capture_output=True,
            check=False,
        )
    else:
        proc.kill()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()

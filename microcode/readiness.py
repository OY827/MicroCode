from __future__ import annotations

from dataclasses import dataclass, field

from microcode.config import FallbackConfig, ModelConfig, load_fallback_config, load_model_config, resolve_protocol

STATUS_ORDER = {"ready": 0, "warning": 1, "blocked": 2}


@dataclass(frozen=True, slots=True)
class ReadinessCheck:
    name: str
    status: str
    detail: str
    next_step: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "next_step": self.next_step,
        }


@dataclass(frozen=True, slots=True)
class ReadinessReport:
    status: str
    summary: str
    protocol: str
    model: str
    checks: list[ReadinessCheck] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "summary": self.summary,
            "protocol": self.protocol,
            "model": self.model,
            "checks": [check.to_dict() for check in self.checks],
        }

    def format(self) -> str:
        lines = [
            f"readiness: {self.status}",
            self.summary,
            f"protocol: {self.protocol}  model: {self.model}",
            "",
        ]
        for check in self.checks:
            lines.append(f"- [{check.status}] {check.name}: {check.detail}")
            if check.next_step:
                lines.append(f"    next: {check.next_step}")
        return "\n".join(lines).rstrip() + "\n"

    def exit_code(self, fail_on: str = "blocked") -> int:
        threshold = fail_on if fail_on in STATUS_ORDER else "blocked"
        if STATUS_ORDER[self.status] >= STATUS_ORDER[threshold]:
            return 2
        return 0


def _url_ok(url: str) -> bool:
    lowered = url.strip().lower()
    return lowered.startswith("http://") or lowered.startswith("https://")


def build_readiness_report(
    *,
    config: ModelConfig | None = None,
    fallback: FallbackConfig | None = None,
) -> ReadinessReport:
    """Local preflight only. Does not call a model provider."""

    primary = config if config is not None else load_model_config()
    spare = fallback if fallback is not None else load_fallback_config()
    protocol = resolve_protocol(primary)
    checks: list[ReadinessCheck] = []

    if primary.is_configured:
        checks.append(
            ReadinessCheck(
                name="primary key",
                status="pass",
                detail="MICROCODE_API_KEY is set.",
            )
        )
    elif spare.is_configured:
        checks.append(
            ReadinessCheck(
                name="primary key",
                status="warning",
                detail="Primary key is missing; a fallback key is set.",
                next_step="Set MICROCODE_API_KEY, or keep using MICROCODE_FALLBACK_API_KEY.",
            )
        )
    else:
        checks.append(
            ReadinessCheck(
                name="primary key",
                status="warning",
                detail="No API key. MicroCode will start the offline mock model.",
                next_step="Set MICROCODE_API_KEY in the environment or a .env file.",
            )
        )

    if primary.model.strip():
        checks.append(
            ReadinessCheck(
                name="model name",
                status="pass",
                detail=f"Using {primary.model}.",
            )
        )
    else:
        checks.append(
            ReadinessCheck(
                name="model name",
                status="blocked",
                detail="Model name is empty.",
                next_step="Set MICROCODE_MODEL.",
            )
        )

    if _url_ok(primary.base_url):
        checks.append(
            ReadinessCheck(
                name="service address",
                status="pass",
                detail=primary.base_url,
            )
        )
    elif primary.is_configured:
        checks.append(
            ReadinessCheck(
                name="service address",
                status="blocked",
                detail=f"Base URL is not http(s): {primary.base_url or '(empty)'}",
                next_step="Set MICROCODE_BASE_URL to an http:// or https:// address.",
            )
        )
    else:
        checks.append(
            ReadinessCheck(
                name="service address",
                status="warning",
                detail="No live key, so the default address is unused.",
            )
        )

    checks.append(
        ReadinessCheck(
            name="request shape",
            status="pass" if protocol in {"anthropic", "openai"} else "blocked",
            detail=f"Calls will use the {protocol} request shape.",
        )
    )

    if spare.is_configured:
        spare_model = spare.model or "(same name as primary unless MICROCODE_FALLBACK_MODEL is set)"
        checks.append(
            ReadinessCheck(
                name="fallback key",
                status="pass",
                detail=f"Spare key is set. Model: {spare_model}.",
            )
        )
    else:
        checks.append(
            ReadinessCheck(
                name="fallback key",
                status="warning",
                detail="No spare key. If the primary path fails, there is no local backup.",
                next_step="Optional: set MICROCODE_FALLBACK_API_KEY.",
            )
        )

    checks.append(
        ReadinessCheck(
            name="live probe",
            status="pass",
            detail="This report is local-only and does not call the model.",
            next_step="Start a turn to see whether the network path actually works.",
        )
    )

    worst = "ready"
    for check in checks:
        mapped = {"pass": "ready", "warning": "warning", "blocked": "blocked"}[check.status]
        if STATUS_ORDER[mapped] > STATUS_ORDER[worst]:
            worst = mapped

    if worst == "ready":
        summary = "Local config looks complete. A live model call has not been tried yet."
    elif worst == "warning":
        summary = "You can start, but a key or fallback is thin. The offline mock may be used."
    else:
        summary = "Do not rely on a live model until the blocked checks are fixed."

    return ReadinessReport(
        status=worst,
        summary=summary,
        protocol=protocol,
        model=primary.model,
        checks=checks,
    )

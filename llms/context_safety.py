"""Deterministic guardrails for untrusted retrieved web content."""

from __future__ import annotations

import re
from dataclasses import dataclass

_INJECTION_RULES = {
    "instruction_override": re.compile(
        r"\b(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|prior|above)\s+"
        r"(?:instructions?|messages?|rules?)\b",
        re.IGNORECASE,
    ),
    "role_impersonation": re.compile(
        r"(?:<\|(?:system|assistant|developer)\|>|\[(?:INST|/INST)\]|"
        r"\b(?:system|developer)\s+message\s*:)",
        re.IGNORECASE,
    ),
    "secret_exfiltration": re.compile(
        r"\b(?:reveal|print|return|show|expose)\b.{0,80}"
        r"\b(?:system\s+prompt|developer\s+message|api\s*key|secret|credential)\b",
        re.IGNORECASE | re.DOTALL,
    ),
}


@dataclass(frozen=True)
class ContextSafetyFinding:
    """Content-free record of one quarantined retrieval result."""

    chunk_id: str
    rule_ids: tuple[str, ...]


@dataclass(frozen=True)
class ContextSafetyResult:
    """Passages safe to present to the model plus quarantine evidence."""

    passages: tuple[dict, ...]
    findings: tuple[ContextSafetyFinding, ...]

    @property
    def filtered_chunk_ids(self) -> tuple[str, ...]:
        return tuple(finding.chunk_id for finding in self.findings)


def filter_untrusted_passages(passages: list[dict]) -> ContextSafetyResult:
    """Quarantine passages matching high-confidence prompt-injection signatures.

    This deliberately targets a small, auditable set of direct instruction attacks.
    It is defense in depth, not a claim that arbitrary prompt injection is solved.
    """

    kept = []
    findings = []
    for index, passage in enumerate(passages, 1):
        title = passage.get("title", "")
        text = passage.get("text", "")
        if not isinstance(title, str) or not isinstance(text, str):
            raise TypeError("retrieved passage title and text must be strings")
        candidate = f"{title}\n{text}"
        matches = tuple(
            rule_id
            for rule_id, pattern in _INJECTION_RULES.items()
            if pattern.search(candidate)
        )
        if matches:
            findings.append(
                ContextSafetyFinding(
                    chunk_id=str(passage.get("chunk_id", f"passage-{index}")),
                    rule_ids=matches,
                )
            )
        else:
            kept.append(passage)
    return ContextSafetyResult(passages=tuple(kept), findings=tuple(findings))

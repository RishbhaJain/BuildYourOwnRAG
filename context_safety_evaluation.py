"""Evaluate deterministic prompt-injection filtering on retrieved passages."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from llms.context_safety import filter_untrusted_passages

VALID_ACTIONS = frozenset({"allow", "filter"})


@dataclass(frozen=True)
class ContextSafetyCase:
    case_id: str
    title: str
    text: str
    expected_action: str
    expected_rule: str | None = None


def load_cases(path: str | Path) -> list[ContextSafetyCase]:
    """Load a validated, uniquely identified JSONL safety suite."""
    cases = []
    seen_ids = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                item = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"line {line_number}: invalid JSON: {exc.msg}"
                ) from exc

            case_id = item.get("id")
            title = item.get("title", "")
            text = item.get("text")
            expected_action = item.get("expected_action")
            expected_rule = item.get("expected_rule")
            if not isinstance(case_id, str) or not case_id.strip():
                raise ValueError(f"line {line_number}: id must be a non-empty string")
            if case_id in seen_ids:
                raise ValueError(f"line {line_number}: duplicate id {case_id!r}")
            if not isinstance(title, str) or not isinstance(text, str) or not text:
                raise ValueError(f"{case_id}: title and non-empty text must be strings")
            if expected_action not in VALID_ACTIONS:
                raise ValueError(f"{case_id}: expected_action must be allow or filter")
            if expected_action == "filter" and not isinstance(expected_rule, str):
                raise ValueError(f"{case_id}: filtered cases require expected_rule")
            if expected_action == "allow" and expected_rule is not None:
                raise ValueError(
                    f"{case_id}: allowed cases cannot declare expected_rule"
                )

            seen_ids.add(case_id)
            cases.append(
                ContextSafetyCase(
                    case_id=case_id,
                    title=title,
                    text=text,
                    expected_action=expected_action,
                    expected_rule=expected_rule,
                )
            )
    if not cases:
        raise ValueError("context safety evaluation requires at least one case")
    return cases


def evaluate(cases: Sequence[ContextSafetyCase]) -> dict[str, object]:
    """Return content-free action and rule-attribution diagnostics."""
    if not cases:
        raise ValueError("context safety evaluation requires at least one case")

    rows = []
    expected_counts = Counter(case.expected_action for case in cases)
    correct_actions = 0
    detected_attacks = 0
    passed_benign = 0
    correct_rules = 0
    rule_counts: dict[str, dict[str, int]] = {}

    for case in cases:
        result = filter_untrusted_passages(
            [{"chunk_id": case.case_id, "title": case.title, "text": case.text}]
        )
        predicted_action = "filter" if result.findings else "allow"
        predicted_rules = result.findings[0].rule_ids if result.findings else ()
        action_correct = predicted_action == case.expected_action
        rule_correct = (
            case.expected_rule is None or case.expected_rule in predicted_rules
        )
        correct_actions += int(action_correct)
        if case.expected_action == "filter":
            detected_attacks += int(predicted_action == "filter")
            correct_rules += int(rule_correct)
            rule_metrics = rule_counts.setdefault(
                case.expected_rule,
                {"cases": 0, "detected": 0, "correctly_attributed": 0},
            )
            rule_metrics["cases"] += 1
            rule_metrics["detected"] += int(predicted_action == "filter")
            rule_metrics["correctly_attributed"] += int(rule_correct)
        else:
            passed_benign += int(predicted_action == "allow")

        rows.append(
            {
                "id": case.case_id,
                "expected_action": case.expected_action,
                "predicted_action": predicted_action,
                "expected_rule": case.expected_rule,
                "predicted_rules": list(predicted_rules),
                "action_correct": action_correct,
                "rule_correct": rule_correct,
            }
        )

    attack_cases = expected_counts["filter"]
    benign_cases = expected_counts["allow"]
    if not attack_cases or not benign_cases:
        raise ValueError("evaluation requires both filter and allow cases")
    return {
        "schema_version": 1,
        "case_count": len(cases),
        "attack_cases": attack_cases,
        "benign_cases": benign_cases,
        "metrics": {
            "action_accuracy": correct_actions / len(cases),
            "attack_recall": detected_attacks / attack_cases,
            "benign_pass_rate": passed_benign / benign_cases,
            "false_positive_rate": 1 - (passed_benign / benign_cases),
            "rule_attribution_accuracy": correct_rules / attack_cases,
        },
        "by_expected_rule": rule_counts,
        "rows": rows,
    }


def parse_threshold(value: str) -> tuple[str, float]:
    try:
        metric, raw_threshold = value.split("=", maxsplit=1)
        threshold = float(raw_threshold)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("threshold must use metric=value") from exc
    if not metric or not 0 <= threshold <= 1:
        raise argparse.ArgumentTypeError("threshold must be between zero and one")
    return metric, threshold


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="JSONL context safety suite")
    parser.add_argument("--output", type=Path, help="optional JSON result artifact")
    parser.add_argument("--minimum", action="append", default=[], type=parse_threshold)
    parser.add_argument("--maximum", action="append", default=[], type=parse_threshold)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = evaluate(load_cases(args.input))
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")

    failures = []
    for direction, thresholds in (("minimum", args.minimum), ("maximum", args.maximum)):
        for metric, threshold in thresholds:
            if metric not in report["metrics"]:
                available = ", ".join(sorted(report["metrics"]))
                raise ValueError(
                    f"unknown metric {metric!r}; available metrics: {available}"
                )
            actual = report["metrics"][metric]
            failed = (
                actual < threshold if direction == "minimum" else actual > threshold
            )
            if failed:
                failures.append(
                    f"{metric}={actual:.4f} violates {direction} {threshold:.4f}"
                )
    if failures:
        print("Context safety regression gate failed: " + "; ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

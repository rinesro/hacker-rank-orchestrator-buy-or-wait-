"""Layer 5 - deterministic validator.  Runs before output.csv is ever written.

Every check here is a hard gate on the output contract.  A violation fails the
run loudly rather than shipping a bad row.  The same checks are reachable
standalone through ``code/evaluation/validate.py`` so a finished ``output.csv``
can be re-verified without re-running the pipeline.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Sequence

from loaders import Dataset, Request
from money import D, q2
from pipeline import COLUMNS

STATUSES = frozenset(
    {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
)
METHODS = frozenset(
    {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}
)
LEGAL_PAIRS = frozenset(
    {
        ("affordable_now", "full_payment"),
        ("affordable_with_plan", "full_payment"),
        ("affordable_with_plan", "partial_payment"),
        ("affordable_with_plan", "installments"),
        ("affordable_later", "wait"),
        ("not_affordable", "not_recommended"),
    }
)
PLAN_ENTRY = re.compile(r"^\d{4}-\d{2}-\d{2}:\d+(?:\.\d{1,2})?$")
CHANGE_ENTRY = re.compile(r"^(?:stop:[A-Za-z0-9_]+|reduce_to:[A-Za-z0-9_]+:\d+(?:\.\d{1,2})?)$")
TOLERANCE = Decimal("0.01")


@dataclass
class Report:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def fail(self, request_id: str, message: str) -> None:
        self.errors.append(f"{request_id}: {message}")

    def warn(self, request_id: str, message: str) -> None:
        self.warnings.append(f"{request_id}: {message}")


def _parse_plan(text: str):
    entries = []
    for chunk in text.split("|"):
        when, _, amount = chunk.partition(":")
        entries.append((date.fromisoformat(when), D(amount)))
    return entries


def validate(rows: Sequence[Dict[str, str]], data: Dataset, requests: Sequence[Request]) -> Report:
    report = Report()

    expected = [r.request_id for r in requests]
    got = [row["request_id"] for row in rows]
    if got != expected:
        report.errors.append(
            f"row order/count mismatch: {len(got)} rows vs {len(expected)} requests"
        )
        return report

    by_id = {r.request_id: r for r in requests}
    for row in rows:
        rid = row["request_id"]
        request = by_id[rid]

        if tuple(row.keys()) != COLUMNS:
            report.fail(rid, "column set or order does not match the contract")

        # --- amount_safe_to_pay -------------------------------------------
        try:
            safe = D(row["amount_safe_to_pay"])
        except InvalidOperation:
            report.fail(rid, f"amount_safe_to_pay is not numeric: {row['amount_safe_to_pay']!r}")
            continue
        if safe < 0 or safe > request.requested_amount + TOLERANCE:
            report.fail(
                rid,
                f"amount_safe_to_pay {safe} outside [0, {request.requested_amount}]",
            )

        # --- enums ---------------------------------------------------------
        status = row["affordability_status"]
        method = row["recommended_payment_method"]
        if status not in STATUSES:
            report.fail(rid, f"unknown affordability_status {status!r}")
        if method not in METHODS:
            report.fail(rid, f"unknown recommended_payment_method {method!r}")
        if (status, method) not in LEGAL_PAIRS:
            report.fail(rid, f"illegal status/method pair ({status}, {method})")

        # --- payment_plan ---------------------------------------------------
        plan_text = row["payment_plan"]
        if method == "not_recommended":
            if plan_text != "none":
                report.fail(rid, "not_recommended must have payment_plan=none")
            plan = []
        else:
            if plan_text == "none":
                report.fail(rid, f"{method} requires a payment_plan")
                plan = []
            else:
                bad = [c for c in plan_text.split("|") if not PLAN_ENTRY.match(c)]
                if bad:
                    report.fail(rid, f"malformed payment_plan entries: {bad}")
                    plan = []
                else:
                    plan = _parse_plan(plan_text)
                    if plan != sorted(plan, key=lambda p: p[0]):
                        report.fail(rid, "payment_plan is not in chronological order")

        # --- earliest_date_for_full_payment ---------------------------------
        earliest = row["earliest_date_for_full_payment"]
        if status == "affordable_now" and earliest != request.request_date.isoformat():
            report.fail(rid, "affordable_now requires earliest == request_date")
        if status == "not_affordable" and earliest:
            report.warn(rid, "not_affordable carries a non-empty earliest date")
        if earliest:
            try:
                date.fromisoformat(earliest)
            except ValueError:
                report.fail(rid, f"earliest_date_for_full_payment not a date: {earliest!r}")

        # --- method-specific plan shape --------------------------------------
        if method == "partial_payment":
            if status != "affordable_with_plan":
                report.fail(rid, "partial_payment requires affordable_with_plan")
            if not request.allows_partial_payment:
                report.fail(rid, "partial_payment on a request that forbids it")
            if len(plan) != 2:
                report.fail(rid, f"partial_payment needs exactly 2 payments, got {len(plan)}")
            else:
                if plan[0][0] != request.request_date:
                    report.fail(rid, "partial_payment first payment must be on request_date")
                if abs(q2(plan[0][1]) - q2(safe)) > TOLERANCE:
                    report.fail(rid, "partial_payment first payment must equal amount_safe_to_pay")
                if earliest and plan[1][0].isoformat() != earliest:
                    report.fail(rid, "partial_payment second payment must be on the earliest date")
                if plan[1][0] > request.desired_completion_date:
                    report.fail(rid, "partial_payment completes after desired_completion_date")
                total = sum((amount for _, amount in plan), Decimal(0))
                if abs(q2(total) - q2(request.requested_amount)) > TOLERANCE:
                    report.fail(rid, f"partial_payment sums to {total}, not {request.requested_amount}")

        if method == "installments":
            options = data.options_for(rid)
            matched = False
            for option in options:
                schedule = [(w, q2(a)) for w, a in option.schedule()]
                if schedule == [(w, q2(a)) for w, a in plan]:
                    matched = True
                    break
            if not matched:
                report.fail(rid, "installment plan does not match any supplied payment option")

        if method == "full_payment" and len(plan) == 1:
            if abs(q2(plan[0][1]) - q2(request.requested_amount)) > TOLERANCE:
                report.fail(rid, "full_payment plan must pay the full requested amount")

        # --- spending_changes_needed -----------------------------------------
        changes_text = row["spending_changes_needed"]
        if changes_text != "none":
            parts = changes_text.split("|")
            if len(parts) > 3:
                report.fail(rid, f"{len(parts)} spending changes, maximum is 3")
            targets: List[str] = []
            profile = data.profiles[request.user_id]
            for part in parts:
                if not CHANGE_ENTRY.match(part):
                    report.fail(rid, f"malformed spending change {part!r}")
                    continue
                bits = part.split(":")
                kind, event_id = bits[0], bits[1]
                targets.append(event_id)
                event = data.events_by_id.get(event_id)
                if event is None:
                    report.fail(rid, f"spending change targets unknown event {event_id}")
                    continue
                if event.user_id != request.user_id:
                    report.fail(rid, f"spending change targets another user's event {event_id}")
                if not event.is_flexible:
                    report.fail(rid, f"spending change targets non-flexible event {event_id}")
                if event.category in profile.categories_to_protect:
                    report.fail(rid, f"spending change targets protected category {event.category}")
                if kind == "stop":
                    if not event.can_stop:
                        report.fail(rid, f"{event_id} is not stoppable")
                    if event.category not in profile.categories_willing_to_stop:
                        report.fail(rid, f"user will not stop {event.category}")
                else:
                    new_amount = D(bits[2])
                    if not event.can_reduce:
                        report.fail(rid, f"{event_id} is not reducible")
                    if event.category not in profile.categories_willing_to_reduce:
                        report.fail(rid, f"user will not reduce {event.category}")
                    if (
                        event.minimum_allowed_amount is not None
                        and new_amount < event.minimum_allowed_amount
                    ):
                        report.fail(
                            rid,
                            f"reduce_to {new_amount} below minimum_allowed_amount "
                            f"{event.minimum_allowed_amount}",
                        )
            if len(set(targets)) != len(targets):
                report.fail(rid, "stop and reduce target the same event")

        if not row["decision_explanation"].strip():
            report.fail(rid, "empty decision_explanation")

    return report


def assert_no_hardcoded_answers(code_dir: str, data: Dataset) -> List[str]:
    """Guard against fitting to individual rows.

    Any request_id or user_id literal appearing in the solution source is an
    automatic disqualification, so the build refuses to proceed if one does.
    """
    ids = {r.request_id for r in data.requests} | {r.request_id for r in data.samples}
    ids |= set(data.profiles)
    pattern = re.compile(r"\b(?:request|user)_\d+\b")
    offenders: List[str] = []
    for root, _, files in os.walk(code_dir):
        if "cache" in root or "__pycache__" in root:
            continue
        for name in sorted(files):
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            with open(path, encoding="utf-8") as handle:
                for number, line in enumerate(handle, 1):
                    for hit in pattern.findall(line):
                        if hit in ids:
                            offenders.append(f"{path}:{number}: hardcoded identifier {hit}")
    return offenders

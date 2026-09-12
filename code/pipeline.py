"""Wiring: dataset + evidence + config -> one output row per request."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from config import DEFAULT, ForecastConfig
from evidence import Evidence, stub_evidence
from explain import render
from loaders import Dataset, Request
from money import fmt_amount, fmt_plan
from planner import Decision, decide
from state import StateBuilder

COLUMNS = (
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
)


def build_row(decision: Decision, cfg: ForecastConfig) -> Dict[str, str]:
    winner = decision.winner

    if winner is None:
        plan = "none"
        changes = "none"
    else:
        plan = "|".join(
            f"{when.isoformat()}:{fmt_plan(amount)}"
            for when, amount in sorted(winner.payments, key=lambda p: p[0])
        )
        changes = "|".join(c.render() for c in winner.changes) or "none"

    earliest = decision.earliest_full_payment
    if earliest is None:
        earliest_text = ""
    elif cfg.blank_earliest_when_not_affordable and decision.status == "not_affordable":
        earliest_text = ""
    else:
        earliest_text = earliest.isoformat()

    return {
        "request_id": decision.request.request_id,
        "amount_safe_to_pay": fmt_amount(decision.amount_safe_to_pay),
        "affordability_status": decision.status,
        "recommended_payment_method": decision.method,
        "payment_plan": plan,
        "earliest_date_for_full_payment": earliest_text,
        "spending_changes_needed": changes,
        "decision_explanation": render(decision),
    }


def run(
    data: Dataset,
    requests: Sequence[Request],
    evidence: Optional[Evidence] = None,
    cfg: ForecastConfig = DEFAULT,
) -> List[Dict[str, str]]:
    """Deterministic: requests are processed in file order, state is rebuilt per row."""
    evidence = evidence or stub_evidence()
    builder = StateBuilder(data, evidence, cfg)
    rows: List[Dict[str, str]] = []
    for request in requests:
        state = builder.build(request.user_id, request.request_date)
        rows.append(build_row(decide(request, state, data, cfg), cfg))
    return rows

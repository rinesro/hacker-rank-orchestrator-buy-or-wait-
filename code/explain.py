"""Layer 6 - deterministic explanation renderer.

The 25 solved samples use a tight template family, and every figure they quote
is already known to the deterministic pipeline: the amount, the dates, the
changed commitments, and ``minimum_balance_to_keep`` (semantics rule S1 - the
"at least X" figure is the user's minimum, not the projected trough, confirmed
on all 11 samples that quote one).

No model is needed here.  Two phrasing splits matter and both are pinned by
the samples:

* ``wait``  - "Pay X in full on D" when the earliest safe date is exactly the
  user's deadline, "Wait until D, then pay X in full" when it is earlier (S4).
* ``not_recommended`` - the "Although X is available today" variant is used
  exactly when the only mechanism the user could have used is partial payment
  (S5, 7/7 samples).
"""

from __future__ import annotations

from typing import Sequence

from money import fmt_date_long, fmt_money
from planner import Decision
from simulator import Change
from state import UserState


def _commitment(state: UserState, change: Change) -> str:
    """Human phrase for one changed commitment, e.g. "the family streaming plan"."""
    for series in state.series:
        if series.series_id == change.series_id:
            return f"the {series.description[0].lower() + series.description[1:]}"
    return "the flexible commitment"


def _changes_clause(state: UserState, changes: Sequence[Change]) -> str:
    currency = state.currency
    parts = []
    for change in changes:
        name = _commitment(state, change)
        if change.kind == "stop":
            parts.append(f"stop {name}")
        else:
            parts.append(f"reduce {name} to {fmt_money(currency, change.new_amount)}")
    if not parts:
        return ""
    if len(parts) == 1:
        clause = parts[0]
    else:
        clause = ", ".join(parts[:-1]) + " and " + parts[-1]
    return clause[0].upper() + clause[1:]


def render(decision: Decision) -> str:
    state = decision.state
    request = decision.request
    currency = state.currency
    minimum = fmt_money(currency, state.minimum_balance)
    winner = decision.winner

    if winner is None:
        if decision.eligible_methods == ("partial_payment",):
            return (
                f"Do not proceed with the {fmt_money(currency, request.requested_amount)} "
                f"request. Although {fmt_money(currency, decision.amount_safe_to_pay)} is "
                "available today, the full amount cannot be completed safely within 90 days."
            )
        return (
            f"Do not make this payment by {fmt_date_long(request.desired_completion_date)}. "
            f"None of the available options keeps the {minimum} minimum protected."
        )

    if winner.method == "wait":
        when, amount = winner.payments[0]
        if when == request.desired_completion_date:
            return (
                f"Pay {fmt_money(currency, amount)} in full on {fmt_date_long(when)}. "
                f"Paying earlier would take the balance below the {minimum} minimum."
            )
        return (
            f"Wait until {fmt_date_long(when)}, then pay {fmt_money(currency, amount)} in "
            f"full. Paying sooner would put the {minimum} minimum at risk."
        )

    if winner.method == "partial_payment":
        (_, first), (second_date, second) = winner.payments
        return (
            f"Pay {fmt_money(currency, first)} today and the remaining "
            f"{fmt_money(currency, second)} on {fmt_date_long(second_date)}. This completes "
            f"the full request and keeps the {minimum} minimum protected."
        )

    if winner.method == "installments":
        when, amount = winner.payments[0]
        return (
            f"Use {len(winner.payments)} installments of {fmt_money(currency, amount)}, "
            f"starting {fmt_date_long(when)}. This leaves at least {minimum} available."
        )

    # full_payment
    _, amount = winner.payments[0]
    if winner.changes:
        clause = _changes_clause(state, winner.changes)
        return (
            f"{clause}, then pay {fmt_money(currency, amount)} today. "
            f"This leaves at least {minimum} available."
        )
    return (
        f"Pay {fmt_money(currency, amount)} today. "
        f"This leaves at least {minimum} available over the next 90 days."
    )

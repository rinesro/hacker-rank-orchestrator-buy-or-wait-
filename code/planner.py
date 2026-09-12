"""Layer 4 - candidate plan generation and ranking.

Enumerates every plan the user could actually be offered, filters by
eligibility, checks each against the simulator, and ranks the survivors in
exactly the order the specification gives:

1. completes the full request by ``desired_completion_date``
2. requires no spending changes
3. lowest total amount paid
4. starts earlier
5. fewer payments
6. lowest ``payment_option_id``

Spending changes are searched only when they can unlock a better-ranked plan,
never speculatively: rank 2 means a safe no-change plan that meets the
deadline can never be beaten by one that adjusts the user's spending.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import List, Optional, Sequence, Tuple

from config import DEFAULT, ForecastConfig
from loaders import Dataset, PaymentOption, Request
from money import ZERO, q2
from simulator import (
    Change,
    Payment,
    earliest_safe_full_payment,
    max_safe_payment_on,
    simulate,
)
from state import Series, UserState

MAX_SPENDING_CHANGES = 3


@dataclass(frozen=True)
class Candidate:
    method: str
    payments: Tuple[Payment, ...]
    changes: Tuple[Change, ...] = ()
    option_id: str = ""

    @property
    def total_paid(self) -> Decimal:
        return sum((amount for _, amount in self.payments), ZERO)

    @property
    def start(self) -> date:
        return min(when for when, _ in self.payments)

    @property
    def last(self) -> date:
        return max(when for when, _ in self.payments)

    @property
    def count(self) -> int:
        return len(self.payments)


@dataclass
class Decision:
    request: Request
    state: UserState
    amount_safe_to_pay: Decimal
    earliest_full_payment: Optional[date]
    winner: Optional[Candidate]
    status: str
    method: str
    eligible_methods: Tuple[str, ...] = ()
    notes: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# eligibility
# --------------------------------------------------------------------------


def installment_months(option: PaymentOption, cfg: ForecastConfig) -> int:
    """Months an installment option spans.

    Every frequency in the dataset is 28, 30 or 31 days, i.e. monthly, so the
    payment count is the month count.  ``span`` is kept as an alternative for
    calibration; the two agree on all six installment samples.
    """
    if cfg.installment_months_rule == "span" and option.payment_frequency_days:
        total = option.number_of_payments * option.payment_frequency_days
        return -(-total // 30)
    return option.number_of_payments


def eligible_option(option: PaymentOption, state: UserState, cfg: ForecastConfig) -> bool:
    profile = state.profile
    if option.payment_method != "installments":
        return False
    if not profile.accepts("installments"):
        return False
    if profile.max_installment_months is None:
        return False
    return installment_months(option, cfg) <= profile.max_installment_months


def eligible_methods(request: Request, state: UserState, data: Dataset, cfg: ForecastConfig):
    """Methods the user could use at all, before any safety check.

    Drives the ``not_recommended`` explanation split (semantics rule S5).
    """
    profile = state.profile
    methods: List[str] = []
    if profile.accepts("full_payment"):
        methods.append("full_payment")
    if request.allows_partial_payment and profile.accepts("partial_payment"):
        methods.append("partial_payment")
    if any(eligible_option(o, state, cfg) for o in data.options_for(request.request_id)):
        methods.append("installments")
    return tuple(methods)


# --------------------------------------------------------------------------
# spending changes
# --------------------------------------------------------------------------


def _change_options(series: Series, state: UserState) -> List[Change]:
    """The permitted adjustments for one flexible series, cheapest impact first."""
    profile = state.profile
    out: List[Change] = []
    anchor = series.anchor_event
    if (
        anchor.can_reduce
        and series.category in profile.categories_willing_to_reduce
        and series.minimum_allowed_amount is not None
        and series.minimum_allowed_amount < series.amount
    ):
        out.append(
            Change(
                kind="reduce_to",
                series_id=series.series_id,
                event_id=anchor.event_id,
                new_amount=series.minimum_allowed_amount,
            )
        )
    if anchor.can_stop and series.category in profile.categories_willing_to_stop:
        out.append(Change(kind="stop", series_id=series.series_id, event_id=anchor.event_id))
    return out


def _saving(series_by_id, change: Change) -> Decimal:
    series = series_by_id.get(change.series_id)
    if series is None:
        return ZERO
    if change.kind == "stop":
        return series.monthly_amount
    return series.monthly_amount - (change.new_amount or ZERO) * series.monthly_amount / series.amount


def change_sets(state: UserState) -> List[Tuple[Change, ...]]:
    """Every permitted combination of up to three changes, least invasive first.

    Stopping and reducing the same event are mutually exclusive, which falls
    out of picking at most one action per series.
    """
    series_by_id = {s.series_id: s for s in state.series}
    per_series = []
    for series in state.flexible_series():
        options = _change_options(series, state)
        if options:
            per_series.append((series.series_id, options))
    if not per_series:
        return []

    sets: List[Tuple[Change, ...]] = []
    for size in range(1, MAX_SPENDING_CHANGES + 1):
        for chosen in itertools.combinations(per_series, size):
            for combo in itertools.product(*[options for _, options in chosen]):
                sets.append(tuple(combo))
    sets.sort(
        key=lambda combo: (
            len(combo),
            sum((_saving(series_by_id, c) for c in combo), ZERO),
            tuple(c.event_id for c in combo),
        )
    )
    return sets


def _first_safe_change_set(
    state: UserState, payments: Sequence[Payment]
) -> Optional[Tuple[Change, ...]]:
    for combo in change_sets(state):
        if simulate(state, payments, combo).safe:
            return combo
    return None


# --------------------------------------------------------------------------
# candidate generation
# --------------------------------------------------------------------------


def build_candidates(
    request: Request,
    state: UserState,
    data: Dataset,
    amount_safe: Decimal,
    earliest: Optional[date],
    cfg: ForecastConfig,
) -> List[Candidate]:
    profile = state.profile
    total = request.requested_amount
    out: List[Candidate] = []

    if profile.accepts("full_payment"):
        out.append(Candidate("full_payment", ((request.request_date, total),)))
        if earliest is not None and earliest > request.request_date:
            out.append(Candidate("wait", ((earliest, total),)))

    if (
        request.allows_partial_payment
        and profile.accepts("partial_payment")
        and ZERO < amount_safe < total
        and earliest is not None
        and earliest <= request.desired_completion_date
        and earliest > request.request_date
    ):
        remainder = total - amount_safe
        out.append(
            Candidate(
                "partial_payment",
                ((request.request_date, amount_safe), (earliest, remainder)),
            )
        )

    for option in data.options_for(request.request_id):
        if not eligible_option(option, state, cfg):
            continue
        schedule = tuple(option.schedule())
        if schedule:
            out.append(Candidate("installments", schedule, option_id=option.payment_option_id))

    return out


def _rank_key(candidate: Candidate, request: Request):
    completes = candidate.last <= request.desired_completion_date
    return (
        0 if completes else 1,
        1 if candidate.changes else 0,
        candidate.total_paid,
        candidate.start,
        candidate.count,
        candidate.option_id or "~",  # options sort before the unnumbered plans
    )


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def decide(
    request: Request,
    state: UserState,
    data: Dataset,
    cfg: ForecastConfig = DEFAULT,
) -> Decision:
    total = request.requested_amount

    # Both measured before any optional spending change (semantics S2, S3).
    amount_safe = max_safe_payment_on(state, request.request_date, cap=total)
    earliest = earliest_safe_full_payment(state, total)

    candidates = build_candidates(request, state, data, amount_safe, earliest, cfg)

    safe_plain = [c for c in candidates if simulate(state, c.payments).safe]
    best_plain = min(safe_plain, key=lambda c: _rank_key(c, request), default=None)

    winner = best_plain
    if best_plain is None or best_plain.last > request.desired_completion_date:
        # Only now can a spending change improve the outcome: it is either the
        # difference between a plan and no plan, or between missing the
        # deadline and meeting it.
        augmented: List[Candidate] = []
        for candidate in candidates:
            if candidate in safe_plain:
                continue
            combo = _first_safe_change_set(state, candidate.payments)
            if combo is not None:
                augmented.append(
                    Candidate(
                        candidate.method, candidate.payments, combo, candidate.option_id
                    )
                )
        pool = safe_plain + augmented
        winner = min(pool, key=lambda c: _rank_key(c, request), default=None)

    status, method = _classify(winner, request, state)
    return Decision(
        request=request,
        state=state,
        amount_safe_to_pay=q2(amount_safe),
        earliest_full_payment=earliest,
        winner=winner,
        status=status,
        method=method,
        eligible_methods=eligible_methods(request, state, data, cfg),
        notes=list(state.notes),
    )


def _classify(candidate: Optional[Candidate], request: Request, state: UserState):
    if candidate is None:
        return "not_affordable", "not_recommended"
    if candidate.method == "wait":
        return "affordable_later", "wait"
    if candidate.method == "full_payment" and not candidate.changes:
        return "affordable_now", "full_payment"
    return "affordable_with_plan", candidate.method

"""Layer 3 - the 90-day safety simulator.  Everything else is built on this.

One primitive: given a candidate set of outgoing payments and an optional set
of spending changes, walk the forecast day by day and report the minimum
projected balance and the first date it breaches ``minimum_balance_to_keep``.

The three quantities the output contract needs all reduce to this function:

* ``amount_safe_to_pay``            -> ``max_safe_payment_on(state, request_date)``
* ``earliest_date_for_full_payment``-> first date ``full`` passes the check
* plan safety                       -> ``simulate(...).safe``

A payment made on day *d* lowers every balance from *d* onwards by exactly its
amount and leaves earlier days untouched, so the minimum balance is piecewise
linear in the payment amount.  ``max_safe_payment_on`` exploits that with a
closed form; ``max_safe_payment_by_search`` re-derives it by bisection, and a
property test asserts the two always agree.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from money import CENT, D, ZERO, q2
from state import Flow, UserState

Payment = Tuple[date, Decimal]


@dataclass(frozen=True)
class Change:
    """A permitted adjustment to one recurring, flexible commitment."""

    kind: str  # "stop" | "reduce_to"
    series_id: str
    event_id: str
    new_amount: Optional[Decimal] = None

    def render(self) -> str:
        from money import fmt_plan

        if self.kind == "stop":
            return f"stop:{self.event_id}"
        return f"reduce_to:{self.event_id}:{fmt_plan(self.new_amount)}"


@dataclass(frozen=True)
class SimResult:
    min_balance: Decimal
    trough_date: date
    breach_date: Optional[date]

    @property
    def safe(self) -> bool:
        return self.breach_date is None


def _scaled_flows(state: UserState, changes: Sequence[Change]) -> List[Flow]:
    """Apply spending changes to the projected flows of the named series."""
    if not changes:
        return state.flows
    by_series: Dict[str, Decimal] = {}
    for item in state.series:
        by_series[item.series_id] = item.amount
    ratios: Dict[str, Decimal] = {}
    for change in changes:
        base = by_series.get(change.series_id)
        if base is None or base == 0:
            continue
        if change.kind == "stop":
            ratios[change.series_id] = ZERO
        elif change.new_amount is not None:
            ratios[change.series_id] = change.new_amount / base

    adjusted: List[Flow] = []
    for flow in state.flows:
        ratio = ratios.get(flow.series_id) if flow.origin in ("recurring", "burn") else None
        if ratio is None:
            adjusted.append(flow)
            continue
        adjusted.append(
            Flow(
                on=flow.on,
                amount=flow.amount * ratio,
                category=flow.category,
                origin=flow.origin,
                event_id=flow.event_id,
                series_id=flow.series_id,
                note=flow.note,
            )
        )
    return adjusted


def daily_balances(
    state: UserState,
    payments: Sequence[Payment] = (),
    changes: Sequence[Change] = (),
    horizon_end: Optional[date] = None,
) -> List[Tuple[date, Decimal]]:
    """Closing balance for every day of the forecast, inclusive of both ends."""
    end = horizon_end or state.horizon_end
    movement: Dict[date, Decimal] = {}
    for flow in _scaled_flows(state, changes):
        if state.as_of <= flow.on <= end:
            movement[flow.on] = movement.get(flow.on, ZERO) + flow.amount
    for when, amount in payments:
        if when <= end:
            key = max(when, state.as_of)
            movement[key] = movement.get(key, ZERO) - amount

    balances: List[Tuple[date, Decimal]] = []
    balance = state.opening_balance
    cursor = state.as_of
    while cursor <= end:
        balance = balance + movement.get(cursor, ZERO)
        balances.append((cursor, balance))
        cursor += timedelta(days=1)
    return balances


def simulate(
    state: UserState,
    payments: Sequence[Payment] = (),
    changes: Sequence[Change] = (),
    horizon_end: Optional[date] = None,
) -> SimResult:
    """Minimum projected balance and the first breach of the minimum, if any."""
    series = daily_balances(state, payments, changes, horizon_end)
    floor = state.minimum_balance
    trough_date, low = series[0]
    breach: Optional[date] = None
    for when, balance in series:
        if balance < low:
            low, trough_date = balance, when
        if breach is None and balance < floor:
            breach = when
    return SimResult(min_balance=low, trough_date=trough_date, breach_date=breach)


def headroom_split(
    state: UserState,
    when: date,
    changes: Sequence[Change] = (),
    horizon_end: Optional[date] = None,
) -> Tuple[Decimal, Decimal]:
    """Minimum balance strictly before ``when``, and from ``when`` onwards.

    A payment on ``when`` only affects the second half, which is what makes
    the safe amount a closed form rather than a search.
    """
    series = daily_balances(state, (), changes, horizon_end)
    pivot = max(when, state.as_of)
    before = [b for d, b in series if d < pivot]
    after = [b for d, b in series if d >= pivot]
    pre = min(before) if before else None
    post = min(after) if after else None
    return (
        pre if pre is not None else Decimal("Infinity"),
        post if post is not None else Decimal("Infinity"),
    )


def max_safe_payment_on(
    state: UserState,
    when: date,
    cap: Optional[Decimal] = None,
    changes: Sequence[Change] = (),
    horizon_end: Optional[date] = None,
) -> Decimal:
    """Largest single payment on ``when`` that keeps the whole forecast safe."""
    pre, post = headroom_split(state, when, changes, horizon_end)
    floor = state.minimum_balance
    if pre < floor:
        return ZERO  # already breached before the payment; no amount helps
    room = post - floor
    if room <= 0:
        return ZERO
    if cap is not None and room > cap:
        room = cap
    return room


def max_safe_payment_by_search(
    state: UserState,
    when: date,
    cap: Decimal,
    changes: Sequence[Change] = (),
    horizon_end: Optional[date] = None,
) -> Decimal:
    """Bisection twin of ``max_safe_payment_on``, kept for property testing."""
    if not simulate(state, [(when, ZERO)], changes, horizon_end).safe:
        return ZERO
    low, high = ZERO, q2(cap)
    if simulate(state, [(when, high)], changes, horizon_end).safe:
        return high
    # invariant: low is safe, high is not
    while high - low > CENT:
        mid = q2((low + high) / 2)
        if mid <= low:
            break
        if simulate(state, [(when, mid)], changes, horizon_end).safe:
            low = mid
        else:
            high = mid
    return low


def earliest_safe_full_payment(
    state: UserState,
    amount: Decimal,
    changes: Sequence[Change] = (),
) -> Optional[date]:
    """First date in the forecast on which one full payment passes the check.

    Measured without optional spending changes and independently of which
    methods the user will consider - it is a capacity measure, not a
    recommendation (``semantics.md`` rule S3).
    """
    cfg = state.config
    cursor = state.as_of
    while cursor <= state.horizon_end:
        end = state.horizon_end
        if cfg.earliest_window == "sliding":
            end = cursor + timedelta(days=cfg.horizon_days)
        if simulate(state, [(cursor, amount)], changes, end).safe:
            return cursor
        cursor += timedelta(days=1)
    return None


def plan_is_safe(
    state: UserState,
    payments: Sequence[Payment],
    changes: Sequence[Change] = (),
) -> bool:
    return simulate(state, payments, changes).safe

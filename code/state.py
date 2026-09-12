"""Layer 2 - financial state reconstruction.  Pure Python, no model calls.

Turns one user's raw event history into the two things the simulator needs:
an opening balance, and a dated list of signed cash flows covering the
forecast horizon.

Cash-state rules, straight from the spec:

* ``settled`` rows are already inside ``current_available_balance`` - the data
  confirms it, since every settled row in the dataset falls on or before its
  user's request_date.  They therefore feed recurrence detection only.
* ``pending`` debits are reserved; ``pending`` credits are ignored.
* ``scheduled`` rows are honoured on their settlement date.
* ``failed`` and ``cancelled`` rows are dropped.
* ``non_cash`` / ``unrealized`` rows never touch cash.
* Duplicates are collapsed through ``linked_event_id`` lifecycles.

Recurrence is claimed only when the history supports a cadence: at least
``min_occurrences`` settled occurrences with a stable interval.  Everything
else is treated as one-off and is not projected forward.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Sequence, Tuple

from config import DEFAULT, ForecastConfig
from evidence import Amendment, AmendmentType, Evidence
from loaders import Dataset, Event, Profile
from money import D, ZERO

DROPPED_STATUSES = frozenset({"failed", "cancelled", "unrealized"})
CASHLESS_DIRECTIONS = frozenset({"non_cash"})


# --------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Flow:
    """One dated, signed cash movement in the user's home currency."""

    on: date
    amount: Decimal  # positive = credit, negative = debit
    category: str
    origin: str  # "explicit" | "recurring" | "burn"
    event_id: str = ""
    series_id: str = ""
    note: str = ""


@dataclass
class Series:
    """A recurring commitment detected from settled history."""

    series_id: str
    user_id: str
    category: str
    direction: str
    anchor_event: Event  # most recent occurrence; the id quoted in output
    amount: Decimal  # home currency, estimator applied
    period_days: int
    monthly_day: Optional[int]  # set when the cadence is monthly
    last_seen: date
    occurrences: int
    flexibility: str
    minimum_allowed_amount: Optional[Decimal]
    description: str

    @property
    def is_debit(self) -> bool:
        return self.direction == "debit"

    @property
    def monthly_amount(self) -> Decimal:
        """Amount normalised to a 30-day month, for ranking spending changes."""
        if self.period_days <= 0:
            return self.amount
        return self.amount * Decimal(30) / Decimal(self.period_days)

    def occurrences_in(self, start: date, end: date) -> List[date]:
        """Projected occurrence dates in [start, end], after the last real one."""
        dates: List[date] = []
        if self.monthly_day is not None:
            cursor = _add_months(self.last_seen, 1, self.monthly_day)
            while cursor <= end:
                if cursor >= start:
                    dates.append(cursor)
                cursor = _add_months(cursor, 1, self.monthly_day)
        else:
            cursor = self.last_seen + timedelta(days=self.period_days)
            while cursor <= end:
                if cursor >= start:
                    dates.append(cursor)
                cursor = cursor + timedelta(days=self.period_days)
        return dates


@dataclass
class UserState:
    """Everything the simulator and planner need about one user at one date."""

    profile: Profile
    as_of: date
    horizon_end: date
    opening_balance: Decimal
    flows: List[Flow]
    series: List[Series]
    config: ForecastConfig
    notes: List[str] = field(default_factory=list)

    @property
    def minimum_balance(self) -> Decimal:
        return self.profile.minimum_balance_to_keep

    @property
    def currency(self) -> str:
        return self.profile.home_currency

    def flexible_series(self) -> List[Series]:
        """Recurring debits the user has permitted changes to, ranked stably."""
        allowed: List[Series] = []
        protect = set(self.profile.categories_to_protect)
        reduce_ok = set(self.profile.categories_willing_to_reduce)
        stop_ok = set(self.profile.categories_willing_to_stop)
        for item in self.series:
            if not item.is_debit or item.category in protect:
                continue
            if item.flexibility == "fixed":
                continue
            can_stop = item.anchor_event.can_stop and item.category in stop_ok
            can_reduce = (
                item.anchor_event.can_reduce
                and item.category in reduce_ok
                and item.minimum_allowed_amount is not None
                and item.minimum_allowed_amount < item.amount
            )
            if can_stop or can_reduce:
                allowed.append(item)
        allowed.sort(key=lambda s: (s.category, s.anchor_event.event_id))
        return allowed


# --------------------------------------------------------------------------
# date helpers
# --------------------------------------------------------------------------


def _add_months(anchor: date, months: int, day: int) -> date:
    month_index = anchor.year * 12 + (anchor.month - 1) + months
    year, month = divmod(month_index, 12)
    month += 1
    last_day = _days_in_month(year, month)
    return date(year, month, min(day, last_day))


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        return 31
    return (date(year, month + 1, 1) - timedelta(days=1)).day


# --------------------------------------------------------------------------
# estimation
# --------------------------------------------------------------------------


def _estimate(values: Sequence[Decimal], how: str) -> Decimal:
    if not values:
        return ZERO
    ordered = list(values)
    if how == "last":
        return ordered[-1]
    if how == "max":
        return max(ordered)
    if how == "min":
        return min(ordered)
    if how == "mean":
        return sum(ordered) / Decimal(len(ordered))
    median = statistics.median(sorted(ordered))
    return D(median)


# --------------------------------------------------------------------------
# builder
# --------------------------------------------------------------------------


class StateBuilder:
    def __init__(self, data: Dataset, evidence: Evidence, config: ForecastConfig = DEFAULT):
        self.data = data
        self.evidence = evidence
        self.config = config

    # -- amounts -----------------------------------------------------------

    def _amount_home(self, event: Event) -> Optional[Decimal]:
        """Home-currency amount, filling blanks from image evidence when present."""
        value = event.amount
        if value is None:
            value = self.evidence.amount_for_event(event.event_id)
            if value is None:
                return None
        return self.data.event_amount_home(event, value)

    # -- classification ----------------------------------------------------

    def _duplicate_ids(self, events: Sequence[Event]) -> set:
        """Pending rows that merely restate an already-settled movement."""
        if not self.config.drop_linked_duplicates:
            return set()
        settled = {
            e.event_id: e for e in events if e.status == "settled"
        }
        duplicates = set()
        for event in events:
            if not event.linked_event_id or event.status != "pending":
                continue
            original = settled.get(event.linked_event_id)
            if original is None:
                continue
            if original.direction == event.direction and original.amount == event.amount:
                duplicates.add(event.event_id)
        return duplicates

    def _explicit_flows(self, events: Sequence[Event], as_of: date, end: date) -> List[Flow]:
        """Pending and scheduled rows that still have to hit the account."""
        flows: List[Flow] = []
        duplicates = self._duplicate_ids(events)
        for event in events:
            if event.status in DROPPED_STATUSES or event.direction in CASHLESS_DIRECTIONS:
                continue
            if event.status == "settled":
                continue  # already inside current_available_balance
            if event.event_id in duplicates:
                continue
            if event.status == "pending":
                if event.direction == "credit" and not self.config.count_pending_credits:
                    continue
                if event.direction == "debit" and not self.config.reserve_pending_debits:
                    continue
            if event.status == "scheduled":
                if event.direction == "credit" and not self.config.count_scheduled_credits:
                    continue
            amount = self._amount_home(event)
            if amount is None:
                continue  # blank amount with no evidence: never treated as zero
            when = max(event.settlement_date, as_of)
            if when > end:
                continue
            signed = amount if event.direction == "credit" else -amount
            flows.append(
                Flow(
                    on=when,
                    amount=signed,
                    category=event.category,
                    origin="explicit",
                    event_id=event.event_id,
                    note=event.description,
                )
            )
        return flows

    # -- recurrence --------------------------------------------------------

    def _detect_series(self, events: Sequence[Event], as_of: date) -> List[Series]:
        cfg = self.config
        groups: Dict[Tuple[str, str], List[Event]] = {}
        earliest = None
        if cfg.history_window_days:
            earliest = as_of - timedelta(days=cfg.history_window_days)
        for event in events:
            if event.status != "settled" or event.direction in CASHLESS_DIRECTIONS:
                continue
            if event.settlement_date > as_of:
                continue
            if earliest and event.settlement_date < earliest:
                continue
            if event.linked_event_id and event.event_type == "refund":
                continue  # reversal of a specific charge, not a recurring credit
            groups.setdefault((event.direction, event.category), []).append(event)

        series: List[Series] = []
        for (direction, category), members in sorted(groups.items()):
            members = sorted(members, key=lambda e: (e.settlement_date, e.event_id))
            if len(members) < cfg.min_occurrences:
                continue
            dates = [e.settlement_date for e in members]
            gaps = [(b - a).days for a, b in zip(dates, dates[1:]) if (b - a).days > 0]
            if len(gaps) < cfg.min_occurrences - 1:
                continue
            median_gap = int(statistics.median(gaps))
            if median_gap <= 0:
                continue
            regular = sum(
                1 for g in gaps if abs(g - median_gap) <= cfg.gap_tolerance * median_gap
            )
            if regular / len(gaps) < cfg.gap_stability:
                continue

            window = members
            if cfg.estimation_occurrences:
                window = members[-cfg.estimation_occurrences:]
            amounts: List[Decimal] = []
            for event in window:
                value = self._amount_home(event)
                if value is not None:
                    amounts.append(value)
            if not amounts:
                continue
            how = cfg.income_estimator if direction == "credit" else cfg.amount_estimator
            amount = _estimate(amounts, how)

            anchor = members[-1]
            monthly_day = None
            if 26 <= median_gap <= 32:
                monthly_day = anchor.settlement_date.day
            series.append(
                Series(
                    series_id=f"{direction}:{category}",
                    user_id=anchor.user_id,
                    category=category,
                    direction=direction,
                    anchor_event=anchor,
                    amount=amount,
                    period_days=median_gap,
                    monthly_day=monthly_day,
                    last_seen=anchor.settlement_date,
                    occurrences=len(members),
                    flexibility=anchor.flexibility,
                    minimum_allowed_amount=anchor.minimum_allowed_amount,
                    description=anchor.description,
                )
            )
        return series

    def _series_flows(
        self, series: Sequence[Series], as_of: date, end: date, explicit: Sequence[Flow]
    ) -> List[Flow]:
        """Project each series forward, without double-counting explicit rows."""
        claimed: Dict[str, List[date]] = {}
        for flow in explicit:
            claimed.setdefault(flow.category, []).append(flow.on)

        flows: List[Flow] = []
        for item in series:
            if self.config.variable_mode == "daily_burn" and item.monthly_day is None:
                days = (end - as_of).days + 1
                per_day = item.amount / Decimal(item.period_days)
                signed = per_day if not item.is_debit else -per_day
                for offset in range(days):
                    flows.append(
                        Flow(
                            on=as_of + timedelta(days=offset),
                            amount=signed,
                            category=item.category,
                            origin="burn",
                            series_id=item.series_id,
                            event_id=item.anchor_event.event_id,
                            note=item.description,
                        )
                    )
                continue
            for when in item.occurrences_in(as_of, end):
                near = claimed.get(item.category, ())
                if any(abs((when - other).days) <= 3 for other in near):
                    continue
                signed = item.amount if not item.is_debit else -item.amount
                flows.append(
                    Flow(
                        on=when,
                        amount=signed,
                        category=item.category,
                        origin="recurring",
                        series_id=item.series_id,
                        event_id=item.anchor_event.event_id,
                        note=item.description,
                    )
                )
        return flows

    # -- amendments --------------------------------------------------------

    def _apply_amendments(
        self,
        amendments: Sequence[Amendment],
        series: List[Series],
        flows: List[Flow],
        as_of: date,
        end: date,
    ) -> Tuple[List[Series], List[Flow], List[str]]:
        """Apply Layer 1 facts, in the spec's conflict order.

        Ordering: explicit cancellation / settlement / amendment first, then
        the newer record from the same source, then settled over estimate,
        then the financially safer reading.  With stubbed evidence this is a
        no-op, but the ordering lives here so Layer 1 slots in without
        touching the simulator.
        """
        if not amendments:
            return series, flows, []

        notes: List[str] = []
        ordered = sorted(
            amendments,
            key=lambda a: (_AMENDMENT_PRIORITY.get(a.amendment_type, 99), a.message_id),
        )
        by_series = {s.series_id: s for s in series}
        income = by_series.get("credit:salary")

        for item in ordered:
            kind = item.amendment_type
            if kind is AmendmentType.EMPLOYMENT_TERMINATED:
                if income is not None:
                    series = [s for s in series if s is not income]
                    flows = [f for f in flows if f.series_id != income.series_id]
                    income = None
                    notes.append(f"{item.message_id}: salary series removed")
            elif kind is AmendmentType.SALARY_AMOUNT_CHANGE and item.amount is not None:
                effective = item.effective_date or as_of
                flows = [
                    (
                        Flow(
                            on=f.on,
                            amount=item.amount,
                            category=f.category,
                            origin=f.origin,
                            event_id=f.event_id,
                            series_id=f.series_id,
                            note=f.note,
                        )
                        if f.series_id == "credit:salary" and f.on >= effective
                        else f
                    )
                    for f in flows
                ]
                notes.append(f"{item.message_id}: salary amount amended")
            elif kind is AmendmentType.SALARY_DATE_SHIFT and item.effective_date:
                moved: List[Flow] = []
                shifted = False
                for f in flows:
                    if f.series_id == "credit:salary" and not shifted and f.on >= as_of:
                        moved.append(
                            Flow(
                                on=item.effective_date,
                                amount=f.amount,
                                category=f.category,
                                origin=f.origin,
                                event_id=f.event_id,
                                series_id=f.series_id,
                                note=f.note,
                            )
                        )
                        shifted = True
                    else:
                        moved.append(f)
                flows = moved
                notes.append(f"{item.message_id}: salary date shifted")
            elif kind is AmendmentType.RECURRING_EXPENSE_PCT_CHANGE and item.percent:
                factor = Decimal(1) + item.percent / Decimal(100)
                target = item.category
                flows = [
                    (
                        Flow(
                            on=f.on,
                            amount=f.amount * factor,
                            category=f.category,
                            origin=f.origin,
                            event_id=f.event_id,
                            series_id=f.series_id,
                            note=f.note,
                        )
                        if f.category == target and f.origin == "recurring"
                        else f
                    )
                    for f in flows
                ]
                notes.append(f"{item.message_id}: {target} adjusted by {item.percent}%")
            elif kind in _CANCELLING_TYPES and item.applies_to_event_id:
                flows = [f for f in flows if f.event_id != item.applies_to_event_id]
                notes.append(f"{item.message_id}: {item.applies_to_event_id} removed")
            elif kind is AmendmentType.EXPENSE_AMOUNT_AMENDED and item.applies_to_event_id:
                if item.amount is not None:
                    flows = [
                        (
                            Flow(
                                on=f.on,
                                amount=-item.amount if f.amount < 0 else item.amount,
                                category=f.category,
                                origin=f.origin,
                                event_id=f.event_id,
                                series_id=f.series_id,
                                note=f.note,
                            )
                            if f.event_id == item.applies_to_event_id
                            else f
                        )
                        for f in flows
                    ]
                    notes.append(f"{item.message_id}: {item.applies_to_event_id} amended")
            elif kind is AmendmentType.EXPENSE_DATE_SHIFT and item.applies_to_event_id:
                if item.effective_date:
                    flows = [
                        (
                            Flow(
                                on=item.effective_date,
                                amount=f.amount,
                                category=f.category,
                                origin=f.origin,
                                event_id=f.event_id,
                                series_id=f.series_id,
                                note=f.note,
                            )
                            if f.event_id == item.applies_to_event_id
                            else f
                        )
                        for f in flows
                    ]
                    notes.append(f"{item.message_id}: {item.applies_to_event_id} moved")
        flows = [f for f in flows if as_of <= f.on <= end]
        return series, flows, notes

    # -- entry point -------------------------------------------------------

    def build(self, user_id: str, as_of: date) -> UserState:
        profile = self.data.profiles[user_id]
        events = self.data.user_events(user_id)
        end = as_of + timedelta(days=self.config.horizon_days)

        explicit = self._explicit_flows(events, as_of, end)
        series = self._detect_series(events, as_of)
        projected = self._series_flows(series, as_of, end, explicit)
        flows = explicit + projected

        amendments = self.evidence.amendments_for(user_id)
        series, flows, notes = self._apply_amendments(amendments, series, flows, as_of, end)

        flows.sort(key=lambda f: (f.on, f.origin, f.event_id, f.category))
        return UserState(
            profile=profile,
            as_of=as_of,
            horizon_end=end,
            opening_balance=profile.current_available_balance,
            flows=flows,
            series=series,
            config=self.config,
            notes=notes,
        )


_AMENDMENT_PRIORITY = {
    AmendmentType.EXPENSE_CANCELLED: 0,
    AmendmentType.SUBSCRIPTION_CANCELLED: 0,
    AmendmentType.EMPLOYMENT_TERMINATED: 0,
    AmendmentType.EXPENSE_AMOUNT_AMENDED: 1,
    AmendmentType.SALARY_AMOUNT_CHANGE: 1,
    AmendmentType.RECURRING_EXPENSE_PCT_CHANGE: 1,
    AmendmentType.SALARY_DATE_SHIFT: 2,
    AmendmentType.EXPENSE_DATE_SHIFT: 2,
    AmendmentType.INCOME_PENDING_NOT_CONFIRMED: 3,
    AmendmentType.INCOME_IS_ONE_OFF: 3,
    AmendmentType.REFUND_PENDING: 3,
    AmendmentType.FX_SETTLEMENT_PENDING: 3,
    AmendmentType.INTERNAL_TRANSFER_NEUTRAL: 4,
    AmendmentType.UNREALIZED_VALUATION_ONLY: 4,
    AmendmentType.SALARY_ONE_OFF_ADJUSTMENT: 4,
}

_CANCELLING_TYPES = frozenset(
    {
        AmendmentType.EXPENSE_CANCELLED,
        AmendmentType.SUBSCRIPTION_CANCELLED,
        AmendmentType.INCOME_PENDING_NOT_CONFIRMED,
        AmendmentType.INCOME_IS_ONE_OFF,
        AmendmentType.REFUND_PENDING,
        AmendmentType.FX_SETTLEMENT_PENDING,
        AmendmentType.INTERNAL_TRANSFER_NEUTRAL,
        AmendmentType.UNREALIZED_VALUATION_ONLY,
    }
)

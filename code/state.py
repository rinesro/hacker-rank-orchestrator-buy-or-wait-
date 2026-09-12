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

TERMINAL_MARKERS = ("final ", "previous ", "last ")
"""A record that names itself as the end of an income stream.

``Final employer payroll`` and ``Previous employer payroll`` both say the
stream they belong to has stopped.  Where a new stream replaced it, that
stream has its own description and is detected on its own merits; where
nothing replaced it, the user has no further income and the forecast must
say so.  The data supports both readings: of the users carrying such a row,
11 have a later income stream and 7 do not.
"""


def _is_terminal(description: str) -> bool:
    lowered = description.strip().lower()
    return any(lowered.startswith(marker) for marker in TERMINAL_MARKERS)


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

    def _cadence(self, members: Sequence[Event]) -> Optional[Tuple[int, int]]:
        """(median gap, regular-gap count) when the dates support a cadence."""
        cfg = self.config
        if len(members) < cfg.min_occurrences:
            return None
        dates = [e.settlement_date for e in members]
        gaps = [(b - a).days for a, b in zip(dates, dates[1:]) if (b - a).days > 0]
        if len(gaps) < cfg.min_occurrences - 1:
            return None
        median_gap = int(statistics.median(gaps))
        if median_gap <= 0:
            return None
        regular = sum(1 for g in gaps if abs(g - median_gap) <= cfg.gap_tolerance * median_gap)
        if regular / len(gaps) < cfg.gap_stability:
            return None
        return median_gap, regular

    def _anchor_on_cadence(self, members: Sequence[Event], median_gap: int) -> bool:
        """Is the most recent occurrence itself on the detected cadence?

        Projection starts from the last occurrence, so a cadence is only usable
        if that occurrence sits on it.  A payroll history of five monthly
        credits followed by a one-off arrears payment and a final net-salary
        line still looks monthly in aggregate, but projecting from the last row
        would place every future salary on the wrong day of the month.  This is
        the test that separates a genuine single series from a category that is
        quietly holding several.
        """
        if len(members) < 2:
            return False
        last_gap = (members[-1].settlement_date - members[-2].settlement_date).days
        return abs(last_gap - median_gap) <= self.config.gap_tolerance * median_gap

    def _make_series(self, members: Sequence[Event], series_id: str, median_gap: int) -> Optional[Series]:
        cfg = self.config
        window = list(members)
        if cfg.estimation_occurrences:
            window = window[-cfg.estimation_occurrences:]
        amounts: List[Decimal] = []
        for event in window:
            value = self._amount_home(event)
            if value is not None:
                amounts.append(value)
        if not amounts:
            return None
        anchor = members[-1]
        how = cfg.income_estimator if anchor.direction == "credit" else cfg.amount_estimator
        monthly_day = anchor.settlement_date.day if 26 <= median_gap <= 32 else None
        return Series(
            series_id=series_id,
            user_id=anchor.user_id,
            category=anchor.category,
            direction=anchor.direction,
            anchor_event=anchor,
            amount=_estimate(amounts, how),
            period_days=median_gap,
            monthly_day=monthly_day,
            last_seen=anchor.settlement_date,
            occurrences=len(members),
            flexibility=anchor.flexibility,
            minimum_allowed_amount=anchor.minimum_allowed_amount,
            description=anchor.description,
        )

    def _detect_series(self, events: Sequence[Event], as_of: date) -> List[Series]:
        """Named commitments cluster by description; variable spend pools by category.

        One category can hold several genuinely distinct commitments - a base
        salary on the 15th alongside a sales commission on the 24th, or a
        monthly payroll alongside a one-off arrears payment.  Pooling those by
        category alone either invents a nonsense cadence or destroys the
        cadence entirely, so a named recurring commitment is clustered by its
        description first.

        High-frequency variable spending is the opposite case: the weekly
        grocery run is one commitment wearing a dozen different merchant names.
        Those descriptions never form a stable cluster of their own, so they
        fall through to a category-level pool.  ``DESCRIPTION_MIN_GAP_DAYS``
        separates the two: a named commitment recurs monthly or slower, while
        variable spend recurs far more often.
        """
        cfg = self.config
        earliest = None
        if cfg.history_window_days:
            earliest = as_of - timedelta(days=cfg.history_window_days)

        usable: List[Event] = []
        for event in events:
            if event.status != "settled" or event.direction in CASHLESS_DIRECTIONS:
                continue
            if event.settlement_date > as_of:
                continue
            if earliest and event.settlement_date < earliest:
                continue
            if event.linked_event_id and event.event_type == "refund":
                continue  # reversal of a specific charge, not a recurring credit
            usable.append(event)

        pooled: Dict[Tuple[str, str], List[Event]] = {}
        for event in usable:
            pooled.setdefault((event.direction, event.category), []).append(event)

        series: List[Series] = []
        for (direction, category), members in sorted(pooled.items()):
            members = sorted(members, key=lambda e: (e.settlement_date, e.event_id))
            cadence = self._cadence(members)
            if cadence is not None and self._anchor_on_cadence(members, cadence[0]):
                built = self._make_series(members, f"{direction}:{category}", cadence[0])
                if built is not None:
                    series.append(built)
                continue

            # The category as a whole has no usable cadence, so it is holding
            # more than one commitment.  Split by description and judge each on
            # its own: a base salary and a sales commission are two streams, and
            # a one-off arrears payment is neither.
            by_description: Dict[str, List[Event]] = {}
            for event in members:
                by_description.setdefault(event.description, []).append(event)
            for description, group in sorted(by_description.items()):
                group = sorted(group, key=lambda e: (e.settlement_date, e.event_id))
                inner = self._cadence(group)
                if inner is None or not self._anchor_on_cadence(group, inner[0]):
                    continue
                built = self._make_series(
                    group, f"{direction}:{category}:{description}", inner[0]
                )
                if built is not None:
                    series.append(built)

        return [
            s
            for s in series
            if not _is_terminal(s.anchor_event.description) and self._still_running(s, as_of)
        ]

    def _still_running(self, item: Series, as_of: date) -> bool:
        """Has the series actually kept going, or has it quietly lapsed?

        A commitment that missed its most recent due date is not evidence of a
        continuing commitment.  A second household income last paid 47 days ago
        on a 31-day cadence has stopped; projecting it forward invents income
        the history no longer supports, which is exactly what the spec forbids.
        The same test protects against resurrecting a cancelled subscription.
        """
        overdue = (as_of - item.last_seen).days
        return overdue <= item.period_days * (1 + self.config.gap_tolerance)

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

        def income_series() -> List[Series]:
            return [s for s in series if s.direction == "credit"]

        def income_flow_ids() -> set:
            return {s.series_id for s in income_series()}

        for item in ordered:
            kind = item.amendment_type
            if kind is AmendmentType.EMPLOYMENT_TERMINATED:
                targets = income_flow_ids()
                if targets:
                    series = [s for s in series if s.series_id not in targets]
                    flows = [f for f in flows if f.series_id not in targets]
                    notes.append(f"{item.message_id}: income series removed")
            elif kind is AmendmentType.SALARY_AMOUNT_CHANGE and item.amount is not None:
                effective = item.effective_date or as_of
                amount = self._to_home(item, as_of)
                targets = income_flow_ids()
                if targets:
                    flows = [
                        (
                            Flow(
                                on=f.on,
                                amount=amount,
                                category=f.category,
                                origin=f.origin,
                                event_id=f.event_id,
                                series_id=f.series_id,
                                note=f.note,
                            )
                            if f.series_id in targets and f.on >= effective
                            else f
                        )
                        for f in flows
                    ]
                    notes.append(f"{item.message_id}: income amount amended")
                else:
                    # A confirmed salary with no detected series is not invented
                    # income: the employer has stated the amount, and often the
                    # date.  Refusing to count it is not the conservative
                    # reading, it is simply a wrong one.
                    created = self._income_from_amendment(item, amount, as_of, end)
                    flows.extend(created)
                    if created:
                        notes.append(f"{item.message_id}: income series created from message")
            elif kind is AmendmentType.SALARY_DATE_SHIFT and item.effective_date:
                targets = income_flow_ids()
                moved: List[Flow] = []
                shifted = False
                for f in flows:
                    if f.series_id in targets and not shifted and f.on >= as_of:
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
                notes.append(f"{item.message_id}: income date shifted")
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

    # -- income the history alone does not carry ---------------------------

    def _to_home(self, item: Amendment, on: date) -> Decimal:
        """Amendment amounts arrive in the message's currency, not the user's."""
        amount = item.amount or ZERO
        home = self.data.profiles[item.user_id].home_currency
        if not item.currency or item.currency == home:
            return amount
        try:
            return self.data.convert(amount, item.currency, home, item.effective_date or on)
        except Exception:
            return amount

    def _income_from_amendment(
        self, item: Amendment, amount: Decimal, as_of: date, end: date
    ) -> List[Flow]:
        """Monthly income built from a confirmed employer statement."""
        if amount <= 0:
            return []
        first = item.effective_date or as_of
        if first < as_of:
            first = _add_months(first, 1, first.day)
        flows: List[Flow] = []
        cursor = first
        while cursor <= end:
            flows.append(
                Flow(
                    on=cursor,
                    amount=amount,
                    category="salary",
                    origin="recurring",
                    series_id="credit:salary:amended",
                    event_id="",
                    note=f"confirmed by {item.message_id}",
                )
            )
            cursor = _add_months(cursor, 1, first.day)
        return flows

    def _extend_scheduled_income(
        self, explicit: Sequence[Flow], series: Sequence[Series], as_of: date, end: date
    ) -> List[Flow]:
        """Repeat a confirmed future salary monthly when history shows no series.

        A user whose history holds a single prior payslip plus one scheduled
        "next confirmed salary" is not a user with one month of income and then
        nothing.  Projecting only the scheduled row makes the balance drift down
        for the rest of the horizon and turns affordable requests into refusals.
        """
        if not self.config.extend_explicit_income:
            return []
        if any(s.direction == "credit" for s in series):
            return []
        incoming = sorted(
            (f for f in explicit if f.amount > 0 and f.on >= as_of), key=lambda f: f.on
        )
        if not incoming:
            return []
        anchor = incoming[-1]
        flows: List[Flow] = []
        cursor = _add_months(anchor.on, 1, anchor.on.day)
        while cursor <= end:
            flows.append(
                Flow(
                    on=cursor,
                    amount=anchor.amount,
                    category=anchor.category,
                    origin="recurring",
                    series_id="credit:salary:scheduled",
                    event_id=anchor.event_id,
                    note=f"{anchor.note} (repeated monthly)",
                )
            )
            cursor = _add_months(cursor, 1, anchor.on.day)
        return flows

    # -- entry point -------------------------------------------------------

    def build(self, user_id: str, as_of: date) -> UserState:
        profile = self.data.profiles[user_id]
        events = self.data.user_events(user_id)
        end = as_of + timedelta(days=self.config.horizon_days)

        explicit = self._explicit_flows(events, as_of, end)
        series = self._detect_series(events, as_of)
        projected = self._series_flows(series, as_of, end, explicit)
        flows = explicit + projected + self._extend_scheduled_income(explicit, series, as_of, end)

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

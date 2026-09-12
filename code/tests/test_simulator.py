"""Layer 3 tests - the simulator is proven before anything is built on it.

Run with:  python3 -m unittest discover -s code/tests -v

Three kinds of test here:

* hand-computed fixtures, where the expected trough and breach date were
  worked out on paper first;
* algebraic properties the simulator must satisfy for the planner's binary
  search and closed form to be interchangeable;
* projection tests for the calendar edge cases (month-end clamping, cadence).
"""

from __future__ import annotations

import os
import sys
import unittest
from datetime import date, timedelta
from decimal import Decimal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from config import DEFAULT, ForecastConfig  # noqa: E402
from loaders import Event, Profile  # noqa: E402
from money import D, ZERO  # noqa: E402
from simulator import (  # noqa: E402
    Change,
    daily_balances,
    earliest_safe_full_payment,
    headroom_split,
    max_safe_payment_by_search,
    max_safe_payment_on,
    simulate,
)
from state import Flow, Series, UserState  # noqa: E402

START = date(2025, 1, 1)


def make_profile(balance="1000", minimum="200", **kwargs) -> Profile:
    defaults = dict(
        user_id="synthetic",
        home_currency="EUR",
        current_available_balance=D(balance),
        minimum_balance_to_keep=D(minimum),
        financial_priorities=(),
        categories_to_protect=("rent",),
        categories_willing_to_reduce=("dining",),
        categories_willing_to_stop=("streaming",),
        payment_methods=("full_payment", "partial_payment", "installments"),
        max_installment_months=12,
    )
    defaults.update(kwargs)
    return Profile(**defaults)


def make_event(event_id="e1", category="streaming", amount="10", flexibility="stoppable",
               minimum_allowed=None, when=START) -> Event:
    return Event(
        event_id=event_id,
        user_id="synthetic",
        event_type="subscription",
        description="Family streaming plan",
        category=category,
        direction="debit",
        amount=D(amount),
        currency="EUR",
        event_date=when,
        settlement_date=when,
        status="settled",
        linked_event_id="",
        flexibility=flexibility,
        minimum_allowed_amount=D(minimum_allowed) if minimum_allowed is not None else None,
    )


def make_state(flows=(), series=(), balance="1000", minimum="200",
               horizon=90, cfg: ForecastConfig = DEFAULT) -> UserState:
    profile = make_profile(balance=balance, minimum=minimum)
    return UserState(
        profile=profile,
        as_of=START,
        horizon_end=START + timedelta(days=horizon),
        opening_balance=profile.current_available_balance,
        flows=list(flows),
        series=list(series),
        config=cfg,
    )


def debit(day: int, amount: str, **kwargs) -> Flow:
    return Flow(
        on=START + timedelta(days=day),
        amount=-D(amount),
        category=kwargs.pop("category", "groceries"),
        origin=kwargs.pop("origin", "recurring"),
        **kwargs,
    )


def credit(day: int, amount: str, **kwargs) -> Flow:
    return Flow(
        on=START + timedelta(days=day),
        amount=D(amount),
        category=kwargs.pop("category", "salary"),
        origin=kwargs.pop("origin", "recurring"),
        **kwargs,
    )


# --------------------------------------------------------------------------
# hand-computed fixtures
# --------------------------------------------------------------------------


class TestHandComputed(unittest.TestCase):
    def test_trough_before_salary(self):
        """1000 opening, -100 on d5, -300 on d10, +500 on d15, -50 on d20.

        Balances: d5 900, d10 600, d15 1100, d20 1050.  Trough is 600 on d10,
        which sits above the 200 minimum, so nothing breaches.
        """
        state = make_state(
            flows=[debit(5, "100"), debit(10, "300"), credit(15, "500"), debit(20, "50")]
        )
        result = simulate(state)
        self.assertEqual(result.min_balance, D("600"))
        self.assertEqual(result.trough_date, START + timedelta(days=10))
        self.assertIsNone(result.breach_date)
        self.assertTrue(result.safe)

    def test_breach_is_the_first_crossing_not_the_trough(self):
        """-850 on d3 (150, breach), +100 on d6 (250), -100 on d9 (150 again).

        The first breach is d3 even though the run-time minimum is reached
        twice; the trough is the later, equal-valued day only if it is lower.
        """
        state = make_state(flows=[debit(3, "850"), credit(6, "100"), debit(9, "100")])
        result = simulate(state)
        self.assertEqual(result.breach_date, START + timedelta(days=3))
        self.assertEqual(result.min_balance, D("150"))
        self.assertEqual(result.trough_date, START + timedelta(days=3))
        self.assertFalse(result.safe)

    def test_same_day_flows_net_before_the_balance_is_recorded(self):
        """A debit and a credit on one day must not create a phantom breach."""
        state = make_state(flows=[debit(4, "900"), credit(4, "900")])
        result = simulate(state)
        self.assertEqual(result.min_balance, D("1000"))
        self.assertIsNone(result.breach_date)

    def test_payment_lowers_only_days_from_the_payment_onwards(self):
        state = make_state(flows=[debit(10, "300"), credit(15, "500")])
        pay_day = START + timedelta(days=12)
        balances = dict(daily_balances(state, [(pay_day, D("100"))]))
        self.assertEqual(balances[START + timedelta(days=11)], D("700"))
        self.assertEqual(balances[pay_day], D("600"))
        self.assertEqual(balances[START + timedelta(days=15)], D("1100"))

    def test_flows_outside_the_horizon_are_ignored(self):
        state = make_state(flows=[debit(95, "900")], horizon=90)
        self.assertTrue(simulate(state).safe)

    def test_horizon_is_inclusive_of_both_ends(self):
        state = make_state(flows=[debit(90, "900")], horizon=90)
        self.assertFalse(simulate(state).safe)
        self.assertEqual(len(daily_balances(state)), 91)


# --------------------------------------------------------------------------
# algebraic properties
# --------------------------------------------------------------------------


SCENARIOS = [
    [debit(5, "100"), debit(10, "300"), credit(15, "500"), debit(20, "50")],
    [debit(1, "50"), debit(2, "50"), debit(3, "50"), credit(30, "400")],
    [credit(14, "900"), debit(7, "600"), debit(21, "120"), debit(60, "200")],
    [debit(0, "10"), credit(45, "1000"), debit(80, "1500")],
    [],
]


class TestProperties(unittest.TestCase):
    def test_minimum_balance_is_linear_in_a_day_zero_payment(self):
        for flows in SCENARIOS:
            state = make_state(flows=flows)
            base = simulate(state).min_balance
            for amount in ("0", "1", "37.51", "300"):
                shifted = simulate(state, [(START, D(amount))]).min_balance
                self.assertEqual(shifted, base - D(amount))

    def test_closed_form_matches_bisection(self):
        """The planner relies on these being interchangeable."""
        for flows in SCENARIOS:
            for minimum in ("0", "200", "700"):
                state = make_state(flows=flows, minimum=minimum)
                for offset in (0, 7, 30):
                    when = START + timedelta(days=offset)
                    cap = D("5000")
                    closed = max_safe_payment_on(state, when, cap=cap)
                    searched = max_safe_payment_by_search(state, when, cap)
                    self.assertLessEqual(
                        abs(closed - searched),
                        D("0.01"),
                        f"flows={flows} min={minimum} day={offset}",
                    )

    def test_safe_amount_is_exactly_safe_and_one_cent_more_is_not(self):
        for flows in SCENARIOS:
            state = make_state(flows=flows, minimum="200")
            amount = max_safe_payment_on(state, START, cap=D("100000"))
            self.assertTrue(simulate(state, [(START, amount)]).safe)
            over = amount + D("0.01")
            self.assertFalse(simulate(state, [(START, over)]).safe)

    def test_safe_amount_never_negative_and_respects_the_cap(self):
        state = make_state(flows=[debit(3, "900")], minimum="200")
        self.assertEqual(max_safe_payment_on(state, START, cap=D("500")), ZERO)
        rich = make_state(flows=[], minimum="200")
        self.assertEqual(max_safe_payment_on(rich, START, cap=D("50")), D("50"))

    def test_headroom_split_partitions_the_forecast(self):
        state = make_state(flows=SCENARIOS[0])
        when = START + timedelta(days=12)
        pre, post = headroom_split(state, when)
        balances = daily_balances(state)
        self.assertEqual(pre, min(b for d, b in balances if d < when))
        self.assertEqual(post, min(b for d, b in balances if d >= when))

    def test_a_breach_before_the_payment_cannot_be_fixed_by_paying_less(self):
        state = make_state(flows=[debit(2, "900"), credit(20, "5000")], minimum="200")
        when = START + timedelta(days=30)
        self.assertEqual(max_safe_payment_on(state, when, cap=D("1000")), ZERO)

    def test_earliest_is_request_date_when_affordable_today(self):
        state = make_state(flows=[], balance="1000", minimum="200")
        self.assertEqual(earliest_safe_full_payment(state, D("500")), START)

    def test_earliest_is_the_first_safe_day_and_nothing_earlier_works(self):
        state = make_state(flows=[credit(20, "900")], balance="1000", minimum="200")
        found = earliest_safe_full_payment(state, D("1200"))
        self.assertEqual(found, START + timedelta(days=20))
        for offset in range(0, 20):
            when = START + timedelta(days=offset)
            self.assertFalse(simulate(state, [(when, D("1200"))]).safe)

    def test_earliest_is_none_when_never_affordable(self):
        state = make_state(flows=[], balance="1000", minimum="200")
        self.assertIsNone(earliest_safe_full_payment(state, D("5000")))


# --------------------------------------------------------------------------
# spending changes
# --------------------------------------------------------------------------


def streaming_series(amount="10", flexibility="stoppable", minimum_allowed=None) -> Series:
    anchor = make_event(amount=amount, flexibility=flexibility, minimum_allowed=minimum_allowed)
    return Series(
        series_id="debit:streaming",
        user_id="synthetic",
        category="streaming",
        direction="debit",
        anchor_event=anchor,
        amount=D(amount),
        period_days=30,
        monthly_day=1,
        last_seen=START,
        occurrences=5,
        flexibility=flexibility,
        minimum_allowed_amount=D(minimum_allowed) if minimum_allowed is not None else None,
        description="Family streaming plan",
    )


class TestSpendingChanges(unittest.TestCase):
    def _state(self, **kwargs):
        flows = [
            Flow(
                on=START + timedelta(days=day),
                amount=-D("10"),
                category="streaming",
                origin="recurring",
                series_id="debit:streaming",
                event_id="e1",
            )
            for day in (10, 40, 70)
        ]
        flows.append(debit(5, "700"))
        return make_state(flows=flows, series=[streaming_series(**kwargs)], minimum="200")

    def test_stop_removes_every_future_occurrence(self):
        state = self._state()
        base = simulate(state).min_balance
        stopped = simulate(state, changes=[Change("stop", "debit:streaming", "e1")]).min_balance
        self.assertEqual(stopped - base, D("30"))

    def test_reduce_to_scales_every_future_occurrence(self):
        state = self._state(flexibility="reducible", minimum_allowed="4")
        base = simulate(state).min_balance
        change = Change("reduce_to", "debit:streaming", "e1", new_amount=D("4"))
        reduced = simulate(state, changes=[change]).min_balance
        self.assertEqual(reduced - base, D("18"))  # 3 occurrences x 6 saved

    def test_changes_never_touch_explicit_or_unrelated_flows(self):
        state = self._state()
        change = Change("stop", "debit:other", "zz")
        self.assertEqual(simulate(state, changes=[change]).min_balance, simulate(state).min_balance)

    def test_change_render_matches_the_output_contract(self):
        self.assertEqual(Change("stop", "s", "event_14").render(), "stop:event_14")
        self.assertEqual(
            Change("reduce_to", "s", "event_21", new_amount=D("100")).render(),
            "reduce_to:event_21:100",
        )
        self.assertEqual(
            Change("reduce_to", "s", "event_21", new_amount=D("23.5")).render(),
            "reduce_to:event_21:23.50",
        )


# --------------------------------------------------------------------------
# recurrence projection
# --------------------------------------------------------------------------


class TestProjection(unittest.TestCase):
    def _monthly(self, last_seen: date, day: int) -> Series:
        series = streaming_series()
        series.last_seen = last_seen
        series.monthly_day = day
        return series

    def test_monthly_cadence_holds_the_day_of_month(self):
        series = self._monthly(date(2025, 1, 15), 15)
        dates = series.occurrences_in(date(2025, 1, 20), date(2025, 4, 30))
        self.assertEqual(dates, [date(2025, 2, 15), date(2025, 3, 15), date(2025, 4, 15)])

    def test_month_end_days_clamp_instead_of_overflowing(self):
        series = self._monthly(date(2025, 1, 31), 31)
        dates = series.occurrences_in(date(2025, 2, 1), date(2025, 5, 1))
        self.assertEqual(dates, [date(2025, 2, 28), date(2025, 3, 31), date(2025, 4, 30)])

    def test_weekly_cadence_steps_by_period_days(self):
        series = streaming_series()
        series.monthly_day = None
        series.period_days = 7
        series.last_seen = date(2025, 1, 1)
        dates = series.occurrences_in(date(2025, 1, 1), date(2025, 1, 31))
        self.assertEqual(
            dates,
            [date(2025, 1, 8), date(2025, 1, 15), date(2025, 1, 22), date(2025, 1, 29)],
        )

    def test_projection_never_emits_the_last_historical_occurrence(self):
        series = self._monthly(date(2025, 1, 15), 15)
        dates = series.occurrences_in(date(2025, 1, 1), date(2025, 2, 1))
        self.assertNotIn(date(2025, 1, 15), dates)


if __name__ == "__main__":
    unittest.main()

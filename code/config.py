"""Every modelling choice the problem statement leaves open, in one place.

The spec says "detect recurrence only when history supports it" and "forecast
essential variable spending conservatively" without defining either.  Rather
than guess once and hope, each open choice is a named switch here, and
``evaluation/calibrate.py`` selects the combination that scores best across
all 25 solved samples at once.

That is model selection over general rules.  No switch may ever be keyed on a
request_id, a user_id or a literal expected answer.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ForecastConfig:
    # --- horizon -----------------------------------------------------------
    horizon_days: int = 90
    """Length of the safety check, counted from request_date inclusive."""

    earliest_window: str = "anchored"
    """``anchored``: a later payment is still only checked to request_date+90.
    ``sliding``: the 90-day check restarts from the payment date."""

    # --- recurrence detection ---------------------------------------------
    min_occurrences: int = 3
    """Minimum settled occurrences before a category counts as recurring."""

    gap_tolerance: float = 0.45
    """A gap counts as regular when it is within this fraction of the median."""

    gap_stability: float = 0.6
    """Fraction of gaps that must be regular for the series to be recurring."""

    history_window_days: int = 0
    """Trailing window used for recurrence detection.  0 = all history."""

    # --- amount estimation -------------------------------------------------
    amount_estimator: str = "median"
    """``median`` | ``mean`` | ``last`` | ``max`` over the estimation window."""

    estimation_occurrences: int = 6
    """How many recent occurrences feed the estimator.  0 = all."""

    variable_mode: str = "discrete"
    """``discrete``: variable spend lands on projected dates.
    ``daily_burn``: variable spend is smeared as a flat daily rate."""

    # --- income ------------------------------------------------------------
    income_estimator: str = "last"
    """``last`` | ``median`` | ``min`` over the estimation window."""

    extend_explicit_income: bool = False
    """Repeat a scheduled salary monthly when history shows no income series."""

    # --- cash-state rules --------------------------------------------------
    reserve_pending_debits: bool = True
    count_pending_credits: bool = False
    count_scheduled_credits: bool = True
    drop_linked_duplicates: bool = True

    # --- output conventions -------------------------------------------------
    blank_earliest_when_not_affordable: bool = True
    """7/7 not_affordable samples leave earliest_date_for_full_payment empty."""

    installment_months_rule: str = "payments"
    """``payments``: months = number_of_payments (all frequencies are monthly).
    ``span``: months = ceil(number_of_payments * frequency_days / 30)."""


DEFAULT = ForecastConfig()

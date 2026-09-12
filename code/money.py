"""Money, rounding and rendering primitives.

Single rounding policy for the whole system: every amount is a ``Decimal``;
quantisation to 2 decimal places with ``ROUND_HALF_UP`` happens only at the
moment a value is rendered into ``output.csv``.  Nothing upstream rounds.

Three distinct output shapes are needed, and the 25 solved samples pin all
three down (see ``evaluation/semantics.md``, rule S6):

``fmt_amount``  ``amount_safe_to_pay``  -> trailing zeros stripped: ``603.3``
``fmt_plan``    ``payment_plan``        -> 2dp when fractional: ``620.40``
``fmt_money``   ``decision_explanation``-> grouped, currency-prefixed: ``EUR 620.40``
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal, ROUND_HALF_UP

CENT = Decimal("0.01")
ZERO = Decimal("0")

_MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


def D(value) -> Decimal:
    """Parse anything into a Decimal without ever going through float."""
    if isinstance(value, Decimal):
        return value
    if value is None:
        return ZERO
    text = str(value).strip().replace(",", "")
    if not text:
        return ZERO
    return Decimal(text)


def q2(value: Decimal) -> Decimal:
    """Quantise to 2dp, ROUND_HALF_UP.  The only rounding in the system."""
    return D(value).quantize(CENT, rounding=ROUND_HALF_UP)


def _digits(value: Decimal) -> str:
    """2dp when the value has a fractional part, plain integer otherwise."""
    rounded = q2(value)
    if rounded == rounded.to_integral_value():
        return str(int(rounded))
    return f"{rounded:.2f}"


def fmt_amount(value: Decimal) -> str:
    """``amount_safe_to_pay`` style: 2dp max, trailing zeros stripped."""
    rounded = q2(value)
    if rounded == rounded.to_integral_value():
        return str(int(rounded))
    return f"{rounded:.2f}".rstrip("0").rstrip(".")


def fmt_plan(value: Decimal) -> str:
    """``payment_plan`` style: integer when integral, otherwise exactly 2dp."""
    return _digits(value)


def fmt_money(currency: str, value: Decimal) -> str:
    """Explanation style: ``ZAR 25,256`` / ``EUR 620.40`` / ``IDR 15,952,906.67``."""
    text = _digits(value)
    if "." in text:
        whole, frac = text.split(".")
        return f"{currency} {int(whole):,}.{frac}"
    return f"{currency} {int(text):,}"


def fmt_date_long(value: date) -> str:
    """Explanation style: ``8 August 2025`` (no leading zero on the day)."""
    return f"{value.day} {_MONTHS[value.month - 1]} {value.year}"


def fmt_date_iso(value: date) -> str:
    return value.isoformat()

"""Layer 0 - deterministic loaders, typed records and currency normalisation.

Reads every file in ``dataset/`` into frozen dataclasses, keyed for O(1) joins.
Money is ``Decimal`` throughout.  Foreign-currency events are converted to the
user's ``home_currency`` using the rate row matched on
``(settlement_date, from_currency, to_currency)``.

Every one of the 140 foreign-currency events in the dataset has a direct rate
row for its own settlement date, so a missing rate means a join bug, not a
data gap: ``convert`` raises rather than falling back to an inverse rate, a
nearby date, or an identity conversion.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from money import D

DATASET_DIRNAME = "dataset"


def repo_root() -> str:
    """Locate the directory holding ``dataset/``, relative to this file.

    Walks upward from this module rather than assuming a fixed depth, so the
    solution runs both in place as ``code/main.py`` and when ``code.zip`` is
    unpacked next to ``dataset/``.  Never a hardcoded path.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    cursor = here
    while True:
        if os.path.isdir(os.path.join(cursor, DATASET_DIRNAME)):
            return cursor
        parent = os.path.dirname(cursor)
        if parent == cursor:
            return os.path.dirname(here)  # fall back to the historical layout
        cursor = parent


def dataset_dir() -> str:
    return os.path.join(repo_root(), DATASET_DIRNAME)


def _date(text: str) -> Optional[date]:
    text = (text or "").strip()
    if not text:
        return None
    return date.fromisoformat(text)


def _opt_decimal(text: str) -> Optional[Decimal]:
    text = (text or "").strip()
    return D(text) if text else None


def _opt_int(text: str) -> Optional[int]:
    text = (text or "").strip()
    return int(text) if text else None


def _pipe_list(text: str) -> Tuple[str, ...]:
    text = (text or "").strip()
    if not text:
        return ()
    return tuple(part.strip() for part in text.split("|") if part.strip())


@dataclass(frozen=True)
class Profile:
    user_id: str
    home_currency: str
    current_available_balance: Decimal
    minimum_balance_to_keep: Decimal
    financial_priorities: Tuple[str, ...]
    categories_to_protect: Tuple[str, ...]
    categories_willing_to_reduce: Tuple[str, ...]
    categories_willing_to_stop: Tuple[str, ...]
    payment_methods: Tuple[str, ...]
    max_installment_months: Optional[int]

    def accepts(self, method: str) -> bool:
        return method in self.payment_methods


@dataclass(frozen=True)
class Event:
    event_id: str
    user_id: str
    event_type: str
    description: str
    category: str
    direction: str
    amount: Optional[Decimal]  # None when the row is blank and needs an image read
    currency: str
    event_date: date
    settlement_date: date
    status: str
    linked_event_id: str
    flexibility: str
    minimum_allowed_amount: Optional[Decimal]

    @property
    def is_flexible(self) -> bool:
        return self.flexibility in ("reducible", "stoppable", "reducible_or_stoppable")

    @property
    def can_stop(self) -> bool:
        return self.flexibility in ("stoppable", "reducible_or_stoppable")

    @property
    def can_reduce(self) -> bool:
        return self.flexibility in ("reducible", "reducible_or_stoppable")


@dataclass(frozen=True)
class Request:
    request_id: str
    user_id: str
    request_date: date
    request_type: str
    requested_amount: Decimal
    desired_completion_date: date
    allows_partial_payment: bool
    request_text: str
    # populated only for sample_requests.csv rows
    truth: Dict[str, str] = field(default_factory=dict)

    @property
    def index(self) -> int:
        """Numeric suffix, used only for stable sorting - never for lookups."""
        return int(self.request_id.rsplit("_", 1)[-1])


@dataclass(frozen=True)
class PaymentOption:
    payment_option_id: str
    request_id: str
    payment_method: str
    payment_amount: Decimal
    number_of_payments: int
    first_payment_date: Optional[date]
    payment_frequency_days: Optional[int]
    financing_fee: Decimal
    total_payable_amount: Decimal

    def schedule(self) -> List[Tuple[date, Decimal]]:
        """Expand the option into explicit (date, amount) payments."""
        if self.first_payment_date is None:
            return []
        if self.number_of_payments <= 1 or not self.payment_frequency_days:
            return [(self.first_payment_date, self.payment_amount)]
        from datetime import timedelta

        step = timedelta(days=self.payment_frequency_days)
        return [
            (self.first_payment_date + step * i, self.payment_amount)
            for i in range(self.number_of_payments)
        ]


@dataclass(frozen=True)
class Message:
    message_id: str
    user_id: str
    request_id: str
    related_event_id: str
    sent_at: str
    source_type: str
    message_text: str


@dataclass(frozen=True)
class ImageRef:
    image_id: str
    user_id: str
    request_id: str
    related_event_id: str

    def path(self) -> str:
        return os.path.join(dataset_dir(), "media", "images", f"{self.image_id}.png")


class MissingRateError(KeyError):
    """Raised when a foreign-currency event has no direct rate for its date."""


class Dataset:
    """All input files, loaded once, with the joins the pipeline needs."""

    def __init__(self, root: Optional[str] = None):
        self.dir = root or dataset_dir()
        self.profiles: Dict[str, Profile] = {}
        self.events_by_user: Dict[str, List[Event]] = {}
        self.events_by_id: Dict[str, Event] = {}
        self.requests: List[Request] = []
        self.samples: List[Request] = []
        self.options_by_request: Dict[str, List[PaymentOption]] = {}
        self.messages_by_user: Dict[str, List[Message]] = {}
        self.messages: List[Message] = []
        self.images: List[ImageRef] = []
        self.images_by_event: Dict[str, ImageRef] = {}
        self.rates: Dict[Tuple[date, str, str], Decimal] = {}
        self._load()

    # ------------------------------------------------------------------ load

    def _rows(self, name: str):
        path = os.path.join(self.dir, name)
        with open(path, newline="", encoding="utf-8") as handle:
            yield from csv.DictReader(handle)

    def _load(self) -> None:
        for row in self._rows("financial_profiles.csv"):
            profile = Profile(
                user_id=row["user_id"],
                home_currency=row["home_currency"],
                current_available_balance=D(row["current_available_balance"]),
                minimum_balance_to_keep=D(row["minimum_balance_to_keep"]),
                financial_priorities=_pipe_list(row["financial_priorities"]),
                categories_to_protect=_pipe_list(row["expense_categories_to_protect"]),
                categories_willing_to_reduce=_pipe_list(
                    row["expense_categories_user_is_willing_to_reduce"]
                ),
                categories_willing_to_stop=_pipe_list(
                    row["expense_categories_user_is_willing_to_stop"]
                ),
                payment_methods=_pipe_list(row["payment_methods_user_will_consider"]),
                max_installment_months=_opt_int(row["max_installment_months"]),
            )
            self.profiles[profile.user_id] = profile

        for row in self._rows("financial_events.csv"):
            event = Event(
                event_id=row["event_id"],
                user_id=row["user_id"],
                event_type=row["event_type"],
                description=row["description"],
                category=row["category"],
                direction=row["direction"],
                amount=_opt_decimal(row["amount"]),
                currency=row["currency"],
                event_date=_date(row["event_date"]),
                settlement_date=_date(row["settlement_date"]) or _date(row["event_date"]),
                status=row["status"],
                linked_event_id=row["linked_event_id"].strip(),
                flexibility=row["flexibility"],
                minimum_allowed_amount=_opt_decimal(row["minimum_allowed_amount"]),
            )
            self.events_by_id[event.event_id] = event
            self.events_by_user.setdefault(event.user_id, []).append(event)
        for events in self.events_by_user.values():
            events.sort(key=lambda e: (e.settlement_date, e.event_date, e.event_id))

        self.requests = [self._request(row) for row in self._rows("requests.csv")]
        self.samples = [
            self._request(row, with_truth=True)
            for row in self._rows("sample_requests.csv")
        ]

        for row in self._rows("request_payment_options.csv"):
            option = PaymentOption(
                payment_option_id=row["payment_option_id"],
                request_id=row["request_id"],
                payment_method=row["payment_method"],
                payment_amount=D(row["payment_amount"]),
                number_of_payments=int(row["number_of_payments"]),
                first_payment_date=_date(row["first_payment_date"]),
                payment_frequency_days=_opt_int(row["payment_frequency_days"]),
                financing_fee=D(row["financing_fee"]),
                total_payable_amount=D(row["total_payable_amount"]),
            )
            self.options_by_request.setdefault(option.request_id, []).append(option)
        for options in self.options_by_request.values():
            options.sort(key=lambda o: o.payment_option_id)

        for row in self._rows("messages.csv"):
            message = Message(
                message_id=row["message_id"],
                user_id=row["user_id"],
                request_id=row["request_id"].strip(),
                related_event_id=row["related_event_id"].strip(),
                sent_at=row["sent_at"],
                source_type=row["source_type"],
                message_text=row["message_text"],
            )
            self.messages.append(message)
            self.messages_by_user.setdefault(message.user_id, []).append(message)
        self.messages.sort(key=lambda m: m.message_id)

        for row in self._rows("images.csv"):
            image = ImageRef(
                image_id=row["image_id"],
                user_id=row["user_id"],
                request_id=row["request_id"].strip(),
                related_event_id=row["related_event_id"].strip(),
            )
            self.images.append(image)
            if image.related_event_id:
                self.images_by_event[image.related_event_id] = image
        self.images.sort(key=lambda i: i.image_id)

        for row in self._rows("exchange_rates.csv"):
            key = (_date(row["rate_date"]), row["from_currency"], row["to_currency"])
            self.rates[key] = D(row["rate"])

    def _request(self, row: dict, with_truth: bool = False) -> Request:
        truth = {}
        if with_truth:
            truth = {
                column: row.get(column, "")
                for column in (
                    "amount_safe_to_pay",
                    "affordability_status",
                    "recommended_payment_method",
                    "payment_plan",
                    "earliest_date_for_full_payment",
                    "spending_changes_needed",
                    "decision_explanation",
                )
            }
        return Request(
            request_id=row["request_id"],
            user_id=row["user_id"],
            request_date=_date(row["request_date"]),
            request_type=row["request_type"],
            requested_amount=D(row["requested_amount"]),
            desired_completion_date=_date(row["desired_completion_date"]),
            allows_partial_payment=row["allows_partial_payment"].strip().lower() == "true",
            request_text=row["request_text"],
            truth=truth,
        )

    # --------------------------------------------------------------- helpers

    def convert(self, amount: Decimal, from_currency: str, to_currency: str, on: date) -> Decimal:
        """Convert using the dated rate row for exactly this pair and date."""
        if from_currency == to_currency:
            return amount
        rate = self.rates.get((on, from_currency, to_currency))
        if rate is None:
            raise MissingRateError(
                f"no {from_currency}->{to_currency} rate for {on.isoformat()}; "
                "every foreign-currency event in this dataset has a direct rate, "
                "so this indicates a join bug rather than missing data"
            )
        return amount * rate

    def event_amount_home(self, event: Event, amount: Optional[Decimal] = None) -> Decimal:
        """Event amount expressed in the user's home currency."""
        value = event.amount if amount is None else amount
        if value is None:
            raise ValueError(f"{event.event_id} has no amount; image evidence required")
        home = self.profiles[event.user_id].home_currency
        return self.convert(value, event.currency, home, event.settlement_date)

    def options_for(self, request_id: str) -> List[PaymentOption]:
        return self.options_by_request.get(request_id, [])

    def user_events(self, user_id: str) -> List[Event]:
        return self.events_by_user.get(user_id, [])

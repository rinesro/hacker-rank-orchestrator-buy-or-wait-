"""Layer 1 boundary - the typed evidence contract, plus a zero-cost stub.

Layer 2 consumes ``Evidence`` and never touches a model, a prompt or an image.
This module defines that contract and a stub implementation that returns no
evidence at all, so the whole deterministic pipeline can be built, tested and
scored before a single token is spent.

The real extractors (vision over the 16 images, batched text over the 215
messages) land behind ``load_evidence`` later and must return exactly these
types.  Anything they cannot map to the closed ``AmendmentType`` enum becomes
``NO_FINANCIAL_IMPACT`` - which is also where prompt-injection attempts go.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional


class AmendmentType(str, Enum):
    """Closed enum.  Extractors may not invent members."""

    SALARY_AMOUNT_CHANGE = "salary_amount_change"
    SALARY_DATE_SHIFT = "salary_date_shift"
    SALARY_ONE_OFF_ADJUSTMENT = "salary_one_off_adjustment"
    EMPLOYMENT_TERMINATED = "employment_terminated"
    INCOME_IS_ONE_OFF = "income_is_one_off"
    INCOME_PENDING_NOT_CONFIRMED = "income_pending_not_confirmed"
    EXPENSE_CANCELLED = "expense_cancelled"
    EXPENSE_AMOUNT_AMENDED = "expense_amount_amended"
    EXPENSE_DATE_SHIFT = "expense_date_shift"
    RECURRING_EXPENSE_PCT_CHANGE = "recurring_expense_pct_change"
    SUBSCRIPTION_CANCELLED = "subscription_cancelled"
    REFUND_PENDING = "refund_pending"
    FX_SETTLEMENT_PENDING = "fx_settlement_pending"
    INTERNAL_TRANSFER_NEUTRAL = "internal_transfer_neutral"
    UNREALIZED_VALUATION_ONLY = "unrealized_valuation_only"
    NO_FINANCIAL_IMPACT = "no_financial_impact"


@dataclass(frozen=True)
class Amendment:
    """One typed fact extracted from one untrusted message."""

    message_id: str
    user_id: str
    amendment_type: AmendmentType
    applies_to_event_id: Optional[str] = None
    effective_date: Optional[date] = None
    amount: Optional[Decimal] = None
    currency: Optional[str] = None
    percent: Optional[Decimal] = None
    category: Optional[str] = None
    recurrence: Optional[str] = None  # "one_off" | "recurring" | None
    confidence: Decimal = Decimal("1")
    quote: str = ""

    @property
    def has_impact(self) -> bool:
        return self.amendment_type is not AmendmentType.NO_FINANCIAL_IMPACT


@dataclass(frozen=True)
class ImageAmount:
    """One amount read off one scanned document."""

    image_id: str
    event_id: str
    amount: Decimal
    currency: str
    document_date: Optional[date] = None
    confidence: Decimal = Decimal("1")
    evidence_snippet: str = ""
    currency_matches_event: bool = True


@dataclass
class Evidence:
    """Everything Layer 1 contributes, indexed for Layer 2."""

    image_amounts: Dict[str, ImageAmount] = field(default_factory=dict)
    amendments_by_user: Dict[str, List[Amendment]] = field(default_factory=dict)

    def amount_for_event(self, event_id: str) -> Optional[Decimal]:
        found = self.image_amounts.get(event_id)
        return found.amount if found else None

    def amendments_for(self, user_id: str) -> List[Amendment]:
        return [a for a in self.amendments_by_user.get(user_id, []) if a.has_impact]

    @property
    def is_empty(self) -> bool:
        return not self.image_amounts and not self.amendments_by_user


def stub_evidence() -> Evidence:
    """Zero AI evidence.  Used for the deterministic baseline."""
    return Evidence()

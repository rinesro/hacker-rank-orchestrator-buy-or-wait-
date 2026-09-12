# Output semantics, reverse-engineered from `sample_requests.csv`

The problem statement leaves several things about the output contract
underdetermined. Every rule below was derived from the 25 solved samples and is
listed with the rows that prove it. Rules the samples cannot settle are listed
at the end as open questions rather than guessed at silently.

Nothing here is keyed on a `request_id` or a `user_id`. `validator.py`
enforces that: `assert_no_hardcoded_answers` walks the source tree and fails
the build if any dataset identifier appears in it.

---

## S1 — the "at least X" figure is `minimum_balance_to_keep`

Every explanation that quotes a currency amount as the balance being protected
quotes the user's own minimum, not the projected trough and not the remaining
balance.

| Row | Explanation figure | `minimum_balance_to_keep` |
|---|---|---|
| request_01 | ZAR 18,000 | 18000 |
| request_02 | IDR 29,158,400 | 29158400 |
| request_06 | EUR 800 | 800 |
| request_07 | INR 93,000 | 93000 |
| request_09 | EUR 600 | 600 |
| request_11 | IDR 34,140,600 | 34140600 |
| request_12 | ZAR 43,200 | 43200 |
| request_16 | INR 122,400 | 122400 |
| request_17 | INR 166,100 | 166100 |
| request_21 | USD 1,800 | 1800 |
| request_22 | EUR 500 | 500 |

11 of 11. This makes the explanation renderer fully deterministic — every
number it needs is already known before any model is consulted.

---

## S2 — `amount_safe_to_pay` is measured *before* optional spending changes

It is the capacity of the unmodified forecast, even when the recommended plan
depends on changing spending.

* **request_06** — safe `603.30`, but the plan pays `620.40` today after
  `stop:event_476`. 620.40 > 603.30, so the figure cannot include the saving.
* **request_21** — safe `1,543.35`, plan pays `1,574.40` after
  `stop:event_1815|reduce_to:event_1816:23.50`.

Corollary used during calibration: in request_06 the shortfall
`620.40 − 603.30 = 17.10` is strictly less than the 19.00/month streaming
subscription being stopped, which pins the trough inside a single streaming
cycle.

---

## S3 — `earliest_date_for_full_payment` ignores the user's method preferences

It is a capacity measure, not a recommendation.

* **request_12** — earliest is `2026-04-05`, the request date itself, yet the
  recommendation is installments starting `2026-04-19`, because user_12 does
  not accept `full_payment`. A preference-aware reading would have left
  earliest blank or moved it.

It is also computed without optional spending changes, consistent with S2.

---

## S4 — the two `wait` phrasings split on whether earliest equals the deadline

| Condition | Template | Rows |
|---|---|---|
| `earliest == desired_completion_date` | "Pay X in full on D. Paying earlier would take the balance below the Y minimum." | request_03, 08, 13, 18, 23 |
| `earliest < desired_completion_date` | "Wait until D, then pay X in full. Paying sooner would put the Y minimum at risk." | request_04 |

5 + 1 of 6. All six wait dates fall on the 15th of a month, which is the
settlement date of the monthly salary in every one of those users' histories —
the wait date is salary-driven, not arbitrary.

---

## S5 — the two `not_recommended` phrasings split on the eligible method set

Variant B ("Do not proceed with the … request. Although X is available today,
the full amount cannot be completed safely within 90 days.") is used **iff the
set of methods the user could have used is exactly `{partial_payment}`** — the
user accepts partial payment, the request allows it, and nothing else is
eligible. Variant A ("Do not make this payment by D. None of the available
options keeps the Y minimum protected.") covers every other case.

| Row | allows partial | methods accepted | eligible set | Variant |
|---|---|---|---|---|
| request_05 | false | full, partial, installments | {full, installments} | A |
| request_10 | true | partial, installments | {partial, installments} | A |
| request_14 | true | partial | **{partial}** | **B** |
| request_15 | false | partial | {} (request forbids partial) | A |
| request_20 | false | full, partial, installments | {full, installments} | A |
| request_24 | true | partial | **{partial}** | **B** |
| request_25 | true | full, installments | {full, installments} | A |

7 of 7. Note request_15: the user accepts only partial payment but the request
forbids it, so the eligible set is empty, not `{partial}` — which is why it
takes variant A. That row is what rules out the simpler "user only accepts
partial" reading.

Implemented in `planner.eligible_methods` and `explain.render`.

---

## S6 — number formatting

| Field | Rule | Evidence |
|---|---|---|
| `payment_plan` | integer when integral, otherwise **exactly** 2dp | `25256`, `122500`, `28820` vs `620.40`, `996.60`, `3246.10`, `1574.40`, `15952906.67` |
| `amount_safe_to_pay` | 2dp maximum, **trailing zeros stripped** | `603.3`, `17229139.2`, `433.4`, `462`, `597.74` |
| explanations | thousands separators, currency-code prefix, same integer/2dp rule | `ZAR 25,256`, `IDR 15,952,906.67`, `EUR 620.40`, `USD 23.50` |
| dates in explanations | day without leading zero, full month name | `8 August 2025`, `15 November 2019`, `12 January 2026` |

The two amount rules genuinely differ: request_06 writes `620.40` in the plan
and `603.3` in `amount_safe_to_pay` in the same row. Single rounding policy
throughout: `Decimal`, `ROUND_HALF_UP`, 2dp, applied only at render time
(`money.py`).

---

## S7 — `max_installment_months` is compared against `number_of_payments`

Every `payment_frequency_days` in the dataset is 28, 30 or 31 — all monthly —
so the payment count is the month count.

| Row | max months | Option taken | Option rejected |
|---|---|---|---|
| request_02 | 7 | 3 × 30d | 18 × 31d |
| request_07 | 12 | 3 × 28d | 15 × 30d |
| request_12 | 11 | 3 × 31d | 21 × 28d |
| request_17 | 3 | 3 × 30d | 18 × 31d |
| request_19 | 2 | **2 × 28d** | **3 × 31d** |
| request_22 | 6 | 3 × 28d | 15 × 30d |

request_19 is the discriminating row: `max_installment_months = 2` accepts the
2-payment option and rejects the 3-payment one. `ceil(n × freq / 30)` agrees on
all six, so the samples cannot separate the two formulas; `number_of_payments`
is used and `installment_months_rule = "span"` is kept in `config.py` as the
documented alternative.

---

## S8 — the ranking order is exactly as specified

* **request_19** proves rank 3 (minimise total paid): partial payment totals
  39,660 against the eligible 2-payment installment option's 41,246.40, and
  partial wins.
* **request_07** proves eligibility gates ranking: installments total 205,296
  against partial's 197,400, and installments still win, because user_07
  accepts `installments` only.
* **request_06** proves rank 1 outranks rank 2: a plan that meets the deadline
  *with* a spending change beats `wait`, which meets no deadline at all
  (earliest `2026-01-15` is after the `2026-01-14` deadline).

---

## S9 — `reduce_to` uses the event's `minimum_allowed_amount`

Not a computed "just enough" reduction.

| Row | Change | `minimum_allowed_amount` on that event |
|---|---|---|
| request_11 | `reduce_to:event_989:665950` | 665950 |
| request_21 | `reduce_to:event_1816:23.50` | 23.5 |

The quoted `event_id` is the **most recent historical occurrence** of the
series, not the first: event_476 is 2025-12-10 against a 2026-01-03 request,
event_989 is 2025-04-23 against 2025-05-03, event_1815/1816 are 2026-03-12 and
2026-03-09 against 2026-04-03.

---

## S10 — `earliest_date_for_full_payment` is blank for `not_affordable`

All 7 `not_affordable` samples leave it empty, which is consistent with the
spec's own definition (the full request cannot be completed safely within the
forecast period, so no such date exists). Controlled by
`blank_earliest_when_not_affordable`, default on.

---

# Forecast model — selected, not assumed

The spec says "detect recurrence only when history supports it" and "forecast
essential variable spending conservatively" without defining either. Each open
choice is a switch in `config.py`, and `evaluation/calibrate.py` selects the
combination scoring best across all 25 samples at once — model selection over
general rules, never per-row fitting.

Selected: `amount_estimator=mean`, `estimation_occurrences=6`,
`income_estimator=last`, `variable_mode=daily_burn`, `gap_tolerance=0.45`
(576 combinations swept; best MAPE 97.95% at stage 1).

## R1 — a category can hold several distinct commitments

Pooling every event in a category into one series either invents a false
cadence or destroys a real one. **80 of 252 users** have salary history spanning
more than one day of the month, and 130 users have two or more distinct income
descriptions.

The rule is *pool first, split only when the pool is incoherent*: try the
category as a whole, and fall back to splitting by description only when the
category has no usable cadence. Splitting by description first was tried and
rejected — it fragments variable spending, giving user_08 a pooled weekly
transport series *plus* three spurious per-merchant series that double-count
the same spend.

## R2 — a cadence is only usable if the latest occurrence sits on it

Projection starts from the last occurrence, so that occurrence must itself be
on-cadence. user_03's salary history is five monthly payroll credits followed
by a one-off arrears payment and a final net-salary line: it still looks
monthly in aggregate (median gap 30) but projecting from the last row places
every future salary on the 31st. This test is what separates a genuine single
series from a category quietly holding several.

## R3 — a series that missed its last due date has lapsed

A commitment whose most recent occurrence is more than `1 + gap_tolerance`
periods old is not evidence of a continuing commitment. user_13's "Second
household income" was last paid 2024-01-20 on a 31-day cadence against a
2024-03-07 request — 47 days, a missed cycle. Projecting it forward invents
income the history no longer supports. Affects **14 of 275 users**, 11 of them
on the income side.

## R4 — a record can name itself as the end of a stream

`Final employer payroll` and `Previous employer payroll` both say the stream
they belong to has stopped. Of the users carrying such a row, 11 have a later
income stream — detected on its own merits, since it has its own description —
and 7 do not, and for those the forecast must show no further income.

This is an events-file signal with no message behind it: user_05 has **no
message at all**, and the only evidence that the salary ended is the word
"Final" in the description.

## R5 — a confirmed salary with no detected series is not invented income

Where an employer message states an amount, and usually a date, refusing to
count it is not the conservative reading — it is a wrong one. user_14's history
is interrupted by unpaid leave and user_15 has only two payslips, so neither
forms a series; both have a message confirming the amount and the credit date.

## R6 — a scheduled salary is repeated monthly when history shows no series

A user with one prior payslip and one scheduled "next confirmed salary" is not
a user with one month of income and then nothing. Without this, request_01 — a
genuinely `affordable_now` row — was reported `not_affordable`.

---

# Open questions the samples do not settle

* **Gig-income streams under a "payout pending" message.** request_10's user
  has four weekly payout streams and a message saying the payout is not
  withdrawable. Counting the streams gives 266,700; suppressing them entirely
  gives 0; the truth is 12,700. Neither reading is right, and no rule stated in
  the data separates them, so no rule was written. Documented rather than
  fitted.
* **Whether `earliest_date_for_full_payment` should be emitted when it exists
  but the status is `not_affordable`.** No sample exercises the case; S10's
  default is applied.
* **`ceil(n × freq / 30)` vs `number_of_payments`** for installment months —
  see S7, the samples agree on both.
* **Explanation paraphrase variety.** The ground truth uses more than one
  phrasing per case: request_01 and request_16 say "This leaves at least X
  available over the next 90 days" while request_09 says "This keeps the X
  minimum available over the next 90 days" for an identical situation. Exact
  string match is therefore unattainable; one deterministic template per case
  is used and measured with token-level F1 (currently 0.839).

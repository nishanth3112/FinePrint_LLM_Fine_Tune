# Phase 1 spec: foundations (no training code)

Budget ~2 hours. Approved with changes; v1 stays lean.

## Goal
A fine-tuned small model turns unstructured advisor/client text into typed constraints.
A deterministic checker uses them to allow/block proposed trades. A human approves each
constraint set. Phase 1 builds schema, checker, eval matcher, securities master. No training.

## Layout
```
src/fineprint/schema.py      constraint types, ConstraintSet, validate_evidence
src/fineprint/portfolio.py   Account, Position, TaxLot, ProposedTrade, SecurityMaster
src/fineprint/themes.py      loader for configs/theme_sic.json
src/fineprint/checker.py     check(), find_conflicts()
src/fineprint/eval/match.py  normalize, match, P/R/F1, bootstrap CIs
configs/theme_sic.json       theme -> SIC approximation (versioned, source URL inside)
scripts/securities_master.py SEC tickers + SIC sample, cached under data/raw/sec/
tests/                       test_schema, test_checker, test_match, test_securities_master
```

## Schema (Pydantic v2, discriminated on `type`, extra="forbid")
Common: `id` (default generated; never scored), `evidence` (required verbatim span; never
matched), ClassVar `cfa_category`. Money is Decimal; weights are fractions in (0,1].
`"5%"` strings are coerced to 0.05; bare numbers are never reinterpreted.

| type | fields | CFA IPS category |
|---|---|---|
| security_restriction | action no_buy/no_sell/no_trade; security (ticker, optional name); until date or None | unique circumstances (legal/regulatory if compliance-driven) |
| sector_esg_exclusion | basis sic/theme; sic_codes; theme; label; applies_to no_buy/no_buy_and_divest | unique circumstances |
| concentration_cap | scope single_security/sector/asset_class; target; max_weight | risk objectives |
| cash_floor | exactly one of min_amount, min_weight | liquidity |
| capital_gains_budget | max_net_realized_gain; tax_year; term short/long/all | tax concerns |
| lot_hold | security; lot_ids or None (all lots); hold_until or None (indefinite); reason | tax concerns |
| recurring_withdrawal | amount; frequency weekly/monthly/quarterly/annually; start_date; end_date | liquidity |
| clarification_needed | question; ambiguity_kind; related_security; candidates | none (meta) |

Sector cap target = SIC major group (2 digits). Theme exclusions resolve through
`configs/theme_sic.json` (tobacco, weapons, fossil_fuels, alcohol, gambling). Every code was
checked against SEC's official SIC list; the table is an approximation (see Limitations).

`ConstraintSet`: constraints, source_hash, status draft/approved, approved_by, approved_at.
Approved requires approver and timestamp.

`validate_evidence(text, cset)` returns a flag per constraint whose evidence is not a
verbatim substring of `text`. Used later as a grounding metric and GRPO reward term.

Each document compiles to its final net constraints. No `lift` type. Cross-document state is
out of scope.

## Checker
`check(cset, account, trades, *, as_of, master=None, theme_table=None,
require_approved=True, withdrawal_horizon_days=30) -> CheckResult`

- Each trade is evaluated independently against the pre-trade account (no running batch
  state). Consistency across checker(gold) vs checker(pred) matters more than realism.
- Deny-overrides: any violated constraint blocks. Every reason is reported.
- `require_approved=True` by default raises NotApprovedError unless status is approved.
  Eval passes False. No other bypass.
- until is inclusive. Past `until` means inactive; the decision carries an EXPIRED note.
- Cap and cash floor use account value, post-trade. Buys only can breach them.
- Gains budget: block a sell if its realized gain > 0 and `realized_gains_ytd + gain` > max.
  The term filter applies to this trade's gain only; ytd is not split by term (approximation).
  Inactive when `tax_year != as_of.year`.
- Lots: specific-ID lots with per-lot cost basis. A sell without lot_ids consumes non-held
  lots FIFO. Wash sales are out of scope. A sell larger than the position, or naming unknown
  lots, is blocked INVALID_TRADE.
- Recurring withdrawals reserve cash for occurrences within `as_of .. as_of+horizon`
  (inclusive), on top of any cash floor. Horizon is a global parameter.
- Open clarification_needed blocks trades in `related_security`; without one it adds a result
  warning.
- `find_conflicts(cset, master, theme_table, as_of)` reports only (a) duplicate caps on the same
  scope+target and (b) lot_hold vs a divest exclusion. `check` includes them in its result.
- Unknown SIC never blocks; it adds an UNCLASSIFIED note.

## Eval matcher
- Key per type (e.g. restriction: type+ticker; cap: type+scope+target; clarification:
  type+related ticker if present). Remaining fields are values. `id`, `evidence`, `label`,
  `reason`, `question`, `candidates` are never scored.
- Pairing: exact (key+values) first, then same-key leftovers are value errors, then
  hallucinated (pred only) and missed (gold only).
- Strict P/R/F1: TP=exact; FP=value errors+hallucinated; FN=value errors+missed.
  Key-level: TP=exact+value errors. Micro-averaged, plus per-type.
- 0/0 convention: both empty gives 1.0; otherwise an undefined side is 0.0.
- Bootstrap: per-example resampling (multinomial weights), 10,000 resamples, fixed seed,
  percentile 95% CI for P, R, F1 (strict and key-level).

## Securities master
Seeded random sample of ~1,500 NYSE+Nasdaq common-stock tickers from
`company_tickers_exchange.json`; SIC fetched per CIK from the submissions API at 5 req/s.
Requires a contact email (`--contact-email` or `SEC_USER_AGENT`); refuses to run without one.
Cached and resumable under `data/raw/sec/` with a manifest (fetched_at, sha256, seed, counts).
Entries without a SIC or non-operating entities are dropped and counted. TODO: bulk
submissions archive.

## Tests (written first; `uv run pytest`)
- schema: round-trip, discriminator/extra rejected, cash_floor exactly-one, weight bounds,
  percent coercion, ticker normalization, theme/sic consistency, approval invariant,
  validate_evidence flags, cfa_category present.
- checker: negations (no_buy/no_sell/no_trade, unmentioned, ticker variants), expired and
  boundary dates, duplicate caps (strictest wins), exclusion vs lot_hold conflict, cap/floor/
  gains-budget boundaries, loss sales, lot hold selective, withdrawal reserve, clarification
  blocking, approval gate, empty set, trades independent of each other.
- match: identical sets, normalization variants, value error vs hallucinated+missed, multiset
  duplicates, empty edge cases, ids/evidence ignored, deterministic bootstrap, CI brackets
  the point estimate.
- securities master: email required, UA header, rate limiting, cache hit, resume, null SIC,
  seeded sampling, exchange/ticker filters.

## Known limitations
- Fund/ETF look-through is deferred. Exclusions and sector caps only see operating companies.
- Theme to SIC is an approximation: SIC is coarse (casinos sit under 7011/7990, which also
  hold non-gambling firms; 2080 covers all beverages; weapons are approximated by 3480/3760).
- Single-document scope; wash sales ignored; realized_gains_ytd is not split by term.
- Master covers a ~1,500-ticker sample, not the full market.

import calendar
from collections import defaultdict
from collections.abc import Mapping
from datetime import date, timedelta
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel

from fineprint.portfolio import Account, Position, ProposedTrade, SecurityMaster, TaxLot
from fineprint.schema import (
    CapitalGainsBudget,
    CashFloor,
    ClarificationNeeded,
    ConcentrationCap,
    ConstraintSet,
    LotHold,
    RecurringWithdrawal,
    SectorESGExclusion,
    SecurityRestriction,
)
from fineprint.themes import load_theme_table

DAYS_FOR_LONG_TERM = 365


class NotApprovedError(RuntimeError):
    pass


class Reason(BaseModel):
    code: str
    constraint_id: str | None = None
    message: str


class TradeDecision(BaseModel):
    trade: ProposedTrade
    decision: Literal["allow", "block"]
    reasons: list[Reason]
    notes: list[Reason]


class Conflict(BaseModel):
    code: str
    constraint_ids: list[str]
    message: str


class CheckResult(BaseModel):
    decisions: list[TradeDecision]
    conflicts: list[Conflict]
    warnings: list[str]


ThemeTable = Mapping[str, frozenset[str]]
Allocation = list[tuple[TaxLot, Decimal]]


def _is_active(until: date | None, as_of: date | None) -> bool:
    return until is None or as_of is None or as_of <= until


def _excluded_codes(ex: SectorESGExclusion, table: ThemeTable) -> frozenset[str]:
    if ex.basis == "sic":
        return frozenset(ex.sic_codes)
    return table.get(ex.theme or "", frozenset())


def _add_months(d: date, months: int) -> date:
    year, month0 = divmod(d.year * 12 + d.month - 1 + months, 12)
    day = min(d.day, calendar.monthrange(year, month0 + 1)[1])
    return date(year, month0 + 1, day)


_STEP_MONTHS = {"monthly": 1, "quarterly": 3, "annually": 12}


def _occurrences(w: RecurringWithdrawal, window_start: date, window_end: date) -> int:
    count, i = 0, 0
    while True:
        if w.frequency == "weekly":
            when = w.start_date + timedelta(weeks=i)
        else:
            when = _add_months(w.start_date, _STEP_MONTHS[w.frequency] * i)
        if when > window_end or (w.end_date is not None and when > w.end_date):
            return count
        if when >= window_start:
            count += 1
        i += 1


def _allocate(lots: list[TaxLot], quantity: Decimal) -> Allocation | None:
    taken: Allocation = []
    remaining = quantity
    for lot in lots:
        if remaining <= 0:
            break
        take = min(lot.quantity, remaining)
        taken.append((lot, take))
        remaining -= take
    return taken if remaining <= 0 else None


class _Evaluator:
    def __init__(
        self,
        cset: ConstraintSet,
        account: Account,
        as_of: date,
        master: SecurityMaster | None,
        table: ThemeTable,
        horizon_days: int,
    ):
        self.cons = cset.constraints
        self.account = account
        self.as_of = as_of
        self.master = master
        self.table = table
        self.window_end = as_of + timedelta(days=horizon_days)
        self.total = account.total_value

    def decide(self, trade: ProposedTrade) -> TradeDecision:
        reasons: list[Reason] = []
        notes: list[Reason] = []

        def block(code: str, c, message: str) -> None:
            reasons.append(Reason(code=code, constraint_id=c.id if c else None, message=message))

        def note(code: str, c, message: str) -> None:
            n = Reason(code=code, constraint_id=c.id, message=message)
            if n not in notes:
                notes.append(n)

        buy = trade.side == "buy"
        post_cash = self.account.cash - trade.value  # only consulted for buys
        floor_required = Decimal(0)
        reserve_by_withdrawal: list[tuple[RecurringWithdrawal, Decimal]] = []

        for c in self.cons:
            if isinstance(c, SecurityRestriction):
                if c.security.ticker != trade.ticker:
                    continue
                if not _is_active(c.until, self.as_of):
                    note("EXPIRED", c, f"restriction ended {c.until}")
                elif c.action == "no_trade" or c.action == f"no_{trade.side}":
                    block("RESTRICTED", c, f"{c.action} on {trade.ticker}")
            elif isinstance(c, ClarificationNeeded):
                related = c.related_security
                if related is not None and related.ticker == trade.ticker:
                    block("UNRESOLVED_CLARIFICATION", c, c.question)
            elif not buy:
                continue
            elif isinstance(c, SectorESGExclusion):
                sic = self._sic(trade.ticker)
                if sic is None:
                    note("UNCLASSIFIED", c, f"no SIC for {trade.ticker}")
                elif sic in _excluded_codes(c, self.table):
                    block("EXCLUDED", c, f"{trade.ticker} (SIC {sic}) is excluded")
            elif isinstance(c, ConcentrationCap):
                self._check_cap(c, trade, block, note)
            elif isinstance(c, CashFloor):
                required = c.min_amount if c.min_amount is not None else (c.min_weight or 0) * self.total
                floor_required = max(floor_required, required)
                if post_cash < required:
                    block("CASH_FLOOR", c, f"cash {post_cash} below floor {required}")
            elif isinstance(c, RecurringWithdrawal):
                n = _occurrences(c, self.as_of, self.window_end)
                if n:
                    reserve_by_withdrawal.append((c, c.amount * n))

        reserve = sum((r for _, r in reserve_by_withdrawal), Decimal(0))
        if buy and reserve and post_cash < floor_required + reserve:
            for w, amount in reserve_by_withdrawal:
                block("WITHDRAWAL_RESERVE", w, f"cash {post_cash} leaves less than {amount} reserved")

        if not buy:
            self._check_sell(trade, block, note)

        return TradeDecision(
            trade=trade, decision="block" if reasons else "allow", reasons=reasons, notes=notes
        )

    def _sic(self, ticker: str) -> str | None:
        return self.master.sic_for(ticker) if self.master else None

    def _check_cap(self, c: ConcentrationCap, trade: ProposedTrade, block, note) -> None:
        def in_scope(p_ticker: str, asset_class: str | None) -> bool | None:
            if c.scope == "single_security":
                return p_ticker == c.target
            if c.scope == "sector":
                sic = self._sic(p_ticker)
                return None if sic is None else sic[:2] == c.target
            return None if asset_class is None else asset_class == c.target

        held = self.account.position(trade.ticker)
        verdict = in_scope(trade.ticker, held.asset_class if held else None)
        if verdict is None:
            note("UNCLASSIFIED", c, f"cannot place {trade.ticker} in {c.scope} {c.target}")
            return
        if not verdict:
            return
        exposure = sum(
            (p.market_value for p in self.account.positions if in_scope(p.ticker, p.asset_class)),
            Decimal(0),
        )
        if exposure + trade.value > c.max_weight * self.total:
            block("CONCENTRATION_CAP", c, f"{c.scope} {c.target} would exceed {c.max_weight}")

    def _check_sell(self, trade: ProposedTrade, block, note) -> None:
        position = self.account.position(trade.ticker)
        if position is None or position.quantity < trade.quantity:
            block("INVALID_TRADE", None, f"cannot sell {trade.quantity} {trade.ticker}")
            return

        holds: list[tuple[LotHold, set[str]]] = []
        for c in self.cons:
            if isinstance(c, LotHold) and c.security.ticker == trade.ticker:
                if _is_active(c.hold_until, self.as_of):
                    ids = {l.lot_id for l in position.lots} if c.lot_ids is None else set(c.lot_ids)
                    holds.append((c, ids))
                else:
                    note("EXPIRED", c, f"hold ended {c.hold_until}")

        allocation = self._allocate_sale(trade, position, holds, block)
        if allocation is None:
            return

        for c in self.cons:
            if not isinstance(c, CapitalGainsBudget):
                continue
            if c.tax_year != self.as_of.year:
                note("INACTIVE_TAX_YEAR", c, f"budget is for {c.tax_year}")
                continue
            gain = sum(
                (
                    (trade.price - lot.cost_basis_per_share) * qty
                    for lot, qty in allocation
                    if c.term == "all" or (c.term == "long") == self._is_long_term(lot)
                ),
                Decimal(0),
            )
            # A loss sale never breaches, even when ytd gains already exceed the budget.
            if gain > 0 and self.account.realized_gains_ytd + gain > c.max_net_realized_gain:
                block("GAINS_BUDGET", c, f"realized gain {gain} exceeds budget")

    def _allocate_sale(self, trade, position: Position, holds, block) -> Allocation | None:
        if trade.lot_ids is not None:
            by_id = {l.lot_id: l for l in position.lots}
            ids = list(dict.fromkeys(trade.lot_ids))
            allocation = (
                _allocate([by_id[i] for i in ids], trade.quantity)
                if all(i in by_id for i in ids)
                else None
            )
            if allocation is None:
                block("INVALID_TRADE", None, f"lots {ids} cannot cover {trade.quantity} {trade.ticker}")
                return None
            used = {lot.lot_id for lot, _ in allocation}
            for c, held in holds:
                if held & used:
                    block("LOT_HOLD", c, f"lot(s) {sorted(held & used)} are on hold")
            return allocation

        fifo = sorted(position.lots, key=lambda l: (l.acquired, l.lot_id))
        held_all = set().union(*(h for _, h in holds)) if holds else set()
        allocation = _allocate([l for l in fifo if l.lot_id not in held_all], trade.quantity)
        if allocation is None:
            for c, held in holds:
                if held:
                    block("LOT_HOLD", c, "unheld lots cannot cover this sale")
            # Still price the sale on all lots so the gains budget is judged too.
            allocation = _allocate(fifo, trade.quantity)
        return allocation

    def _is_long_term(self, lot: TaxLot) -> bool:
        return (self.as_of - lot.acquired).days > DAYS_FOR_LONG_TERM


def find_conflicts(
    cset: ConstraintSet,
    master: SecurityMaster | None = None,
    theme_table: ThemeTable | None = None,
    as_of: date | None = None,
) -> list[Conflict]:
    found: list[Conflict] = []

    caps: dict[tuple[str, str | None], list[ConcentrationCap]] = defaultdict(list)
    for c in cset.constraints:
        if isinstance(c, ConcentrationCap):
            caps[(c.scope, c.target)].append(c)
    for (scope, target), group in caps.items():
        if len(group) > 1:
            found.append(
                Conflict(
                    code="DUPLICATE_CAP",
                    constraint_ids=[g.id for g in group],
                    message=f"{len(group)} caps on {scope} {target}",
                )
            )

    table = theme_table if theme_table is not None else load_theme_table()
    divests = [
        c for c in cset.constraints
        if isinstance(c, SectorESGExclusion) and c.applies_to == "no_buy_and_divest"
    ]
    for hold in cset.constraints:
        if not isinstance(hold, LotHold) or not _is_active(hold.hold_until, as_of) or master is None:
            continue
        sic = master.sic_for(hold.security.ticker)
        for ex in divests:
            if sic is not None and sic in _excluded_codes(ex, table):
                found.append(
                    Conflict(
                        code="LOT_HOLD_VS_DIVEST",
                        constraint_ids=[hold.id, ex.id],
                        message=f"{hold.security.ticker} is held but its sector must be divested",
                    )
                )
    return found


def check(
    cset: ConstraintSet,
    account: Account,
    trades: list[ProposedTrade],
    *,
    as_of: date,
    master: SecurityMaster | None = None,
    theme_table: ThemeTable | None = None,
    require_approved: bool = True,
    withdrawal_horizon_days: int = 30,
) -> CheckResult:
    """Each trade is judged alone against the pre-trade account (no running batch state)."""
    if require_approved and cset.status != "approved":
        raise NotApprovedError("constraint set must be human-approved before checking trades")
    table = theme_table if theme_table is not None else load_theme_table()
    evaluator = _Evaluator(cset, account, as_of, master, table, withdrawal_horizon_days)
    warnings = [
        f"unresolved clarification {c.id}: {c.question}"
        for c in cset.constraints
        if isinstance(c, ClarificationNeeded) and c.related_security is None
    ]
    return CheckResult(
        decisions=[evaluator.decide(t) for t in trades],
        conflicts=find_conflicts(cset, master, table, as_of),
        warnings=warnings,
    )

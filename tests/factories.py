from datetime import UTC, date, datetime
from decimal import Decimal

from fineprint.portfolio import (
    Account,
    DictSecurityMaster,
    Position,
    ProposedTrade,
    TaxLot,
)
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

AS_OF = date(2026, 10, 15)
EV = "evidence text"


def _kw(kw):
    return {"evidence": EV, **kw}


def restriction(ticker="AAPL", action="no_trade", until=None, **kw):
    return SecurityRestriction(
        action=action, security={"ticker": ticker}, until=until, **_kw(kw)
    )


def exclusion(basis="theme", theme="tobacco", sic_codes=(), applies_to="no_buy", **kw):
    return SectorESGExclusion(
        basis=basis,
        theme=theme if basis == "theme" else None,
        sic_codes=list(sic_codes),
        applies_to=applies_to,
        **_kw(kw)
    )


def cap(scope="single_security", target="AAPL", max_weight="0.5", **kw):
    return ConcentrationCap(
        scope=scope, target=target, max_weight=max_weight, **_kw(kw)
    )


def floor(min_amount=None, min_weight=None, **kw):
    return CashFloor(min_amount=min_amount, min_weight=min_weight, **_kw(kw))


def gains_budget(max_gain="1000", tax_year=2026, term="all", **kw):
    return CapitalGainsBudget(
        max_net_realized_gain=max_gain, tax_year=tax_year, term=term, **_kw(kw)
    )


def lot_hold(ticker="AAPL", lot_ids=None, hold_until=None, **kw):
    return LotHold(
        security={"ticker": ticker},
        lot_ids=lot_ids,
        hold_until=hold_until,
        **_kw(kw)
    )


def withdrawal(amount="3000", frequency="monthly", start=date(2026, 1, 20), end=None, **kw):
    return RecurringWithdrawal(
        amount=amount, frequency=frequency, start_date=start, end_date=end, **_kw(kw)
    )


def clarification(related=None, kind="ambiguous_security", **kw):
    return ClarificationNeeded(
        question="Which share class?",
        ambiguity_kind=kind,
        related_security={"ticker": related} if related else None,
        **_kw(kw)
    )


def cset(*constraints, approved=True):
    if approved:
        return ConstraintSet(
            constraints=list(constraints),
            status="approved",
            approved_by="advisor@example.com",
            approved_at=datetime(2026, 10, 14, tzinfo=UTC),
        )
    return ConstraintSet(constraints=list(constraints))


def trade(side="buy", ticker="AAPL", quantity="10", price="100", lot_ids=None):
    return ProposedTrade(
        ticker=ticker, side=side, quantity=quantity, price=price, lot_ids=lot_ids
    )


def make_account(realized_gains_ytd="0", cash="10000"):
    """Total value 40,000: cash 10k, AAPL 10k, MSFT 20k."""
    return Account(
        cash=Decimal(cash),
        realized_gains_ytd=Decimal(realized_gains_ytd),
        positions=[
            Position(
                ticker="AAPL",
                price=Decimal(100),
                asset_class="equity",
                lots=[
                    TaxLot(lot_id="L1", quantity=Decimal(50), cost_basis_per_share=Decimal(40),
                           acquired=date(2020, 1, 1)),
                    TaxLot(lot_id="L2", quantity=Decimal(50), cost_basis_per_share=Decimal(120),
                           acquired=date(2026, 6, 1)),
                ],
            ),
            Position(
                ticker="MSFT",
                price=Decimal(200),
                asset_class="equity",
                lots=[
                    TaxLot(lot_id="M1", quantity=Decimal(80), cost_basis_per_share=Decimal(150),
                           acquired=date(2024, 1, 1)),
                    TaxLot(lot_id="M2", quantity=Decimal(20), cost_basis_per_share=Decimal(100),
                           acquired=date(2026, 8, 1)),
                ],
            ),
        ],
    )


def make_master():
    return DictSecurityMaster({"AAPL": "3571", "MSFT": "7372", "XOM": "1311", "MO": "2111"})

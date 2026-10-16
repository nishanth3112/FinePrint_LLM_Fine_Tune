from datetime import date

import pytest
from factories import (
    AS_OF,
    cap,
    clarification,
    cset,
    exclusion,
    floor,
    gains_budget,
    lot_hold,
    make_account,
    make_master,
    restriction,
    trade,
    withdrawal,
)

from fineprint.checker import NotApprovedError, check, find_conflicts


def run(constraints, trades, account=None, **kw):
    kw.setdefault("master", make_master())
    result = check(
        cset(*constraints), account or make_account(), trades, as_of=AS_OF, **kw
    )
    return result.decisions


def one(constraints, t, **kw):
    return run(constraints, [t], **kw)[0]


def codes(d):
    return {r.code for r in d.reasons}


def note_codes(d):
    return {n.code for n in d.notes}


# ---- security restrictions: negations, tickers -------------------------------------------

def test_no_sell_blocks_sell_only():
    c = [restriction(action="no_sell")]
    assert one(c, trade("sell", lot_ids=["L1"])).decision == "block"
    assert one(c, trade("buy")).decision == "allow"


def test_no_buy_blocks_buy_only():
    c = [restriction(action="no_buy")]
    assert one(c, trade("buy")).decision == "block"
    assert one(c, trade("sell", lot_ids=["L1"])).decision == "allow"


def test_no_trade_blocks_both():
    c = [restriction(action="no_trade")]
    assert one(c, trade("buy")).decision == "block"
    assert one(c, trade("sell", lot_ids=["L1"])).decision == "block"
    assert codes(one(c, trade("buy"))) == {"RESTRICTED"}


def test_unmentioned_security_allowed():
    assert one([restriction("AAPL")], trade("buy", "MSFT")).decision == "allow"


def test_ticker_variants_match():
    c = [restriction("BRK-B")]
    assert one(c, trade("buy", "brk.b")).decision == "block"
    assert one(c, trade("buy", "BRK/B")).decision == "block"


# ---- until-dates -------------------------------------------------------------------------

def test_expired_until_allows_with_note():
    d = one([restriction(until=date(2026, 10, 14))], trade("buy"))
    assert d.decision == "allow" and "EXPIRED" in note_codes(d)


@pytest.mark.parametrize("until", [AS_OF, date(2027, 1, 1), None])
def test_until_inclusive_future_and_open_ended_block(until):
    assert one([restriction(until=until)], trade("buy")).decision == "block"


# ---- concentration caps ------------------------------------------------------------------

def test_cap_boundary():
    c = [cap(max_weight="0.5")]
    assert one(c, trade("buy", quantity="100")).decision == "allow"  # exactly 0.5
    d = one(c, trade("buy", quantity="100.0001"))
    assert d.decision == "block" and codes(d) == {"CONCENTRATION_CAP"}


def test_sells_never_breach_cap():
    assert one([cap(max_weight="0.1")], trade("sell", lot_ids=["L1"])).decision == "allow"


def test_duplicate_caps_strictest_wins_and_conflict_reported():
    loose, tight = cap(max_weight="0.5"), cap(max_weight="0.3")
    result = check(
        cset(loose, tight), make_account(), [trade("buy", quantity="50")],
        as_of=AS_OF, master=make_master(),
    )
    d = result.decisions[0]
    assert d.decision == "block"
    assert {r.constraint_id for r in d.reasons} == {tight.id}
    assert [(c.code, set(c.constraint_ids)) for c in result.conflicts] == [
        ("DUPLICATE_CAP", {loose.id, tight.id})
    ]


def test_sector_cap_uses_sic_major_group():
    c = [cap(scope="sector", target="35", max_weight="0.25")]  # AAPL (3571) already 0.25
    assert one(c, trade("buy", "AAPL")).decision == "block"
    assert one(c, trade("buy", "MSFT")).decision == "allow"


def test_asset_class_cap():
    c = [cap(scope="asset_class", target="equity", max_weight="0.8")]  # equity at 0.75
    assert one(c, trade("buy", quantity="30")).decision == "block"
    assert one(c, trade("buy", quantity="10")).decision == "allow"


# ---- cash floor / withdrawals ------------------------------------------------------------

@pytest.mark.parametrize("c", [floor(min_amount="8000"), floor(min_weight="0.2")], ids=["amt", "wt"])
def test_cash_floor_boundary(c):
    assert one([c], trade("buy", quantity="20")).decision == "allow"  # leaves exactly 8,000
    d = one([c], trade("buy", quantity="20.01"))
    assert d.decision == "block" and codes(d) == {"CASH_FLOOR"}


def test_sell_allowed_even_below_floor():
    assert one([floor(min_amount="50000")], trade("sell", lot_ids=["L1"])).decision == "allow"


def test_withdrawal_reserve_boundary():
    c = [withdrawal(amount="3000")]  # next occurrence 2026-10-20
    assert one(c, trade("buy", quantity="70")).decision == "allow"  # leaves exactly 3,000
    d = one(c, trade("buy", quantity="70.01"))
    assert d.decision == "block" and codes(d) == {"WITHDRAWAL_RESERVE"}


def test_withdrawal_reserve_stacks_on_floor():
    c = [floor(min_amount="1000"), withdrawal(amount="3000")]
    assert one(c, trade("buy", quantity="60")).decision == "allow"
    assert one(c, trade("buy", quantity="60.01")).decision == "block"


@pytest.mark.parametrize(
    "kw",
[{"start": date(2026, 12, 1)}, {"start": date(2026, 1, 1), "end": date(2026, 9, 30)}],
    ids=["starts_after_window", "ended_before"],
)
def test_withdrawal_outside_window_reserves_nothing(kw):
    assert one([withdrawal(**kw)], trade("buy", quantity="100")).decision == "allow"


def test_withdrawal_horizon_is_inclusive_and_configurable():
    c = [withdrawal(amount="3000")]  # occurrence on 2026-10-20 = as_of + 5
    t = trade("buy", quantity="100")
    assert one(c, t, withdrawal_horizon_days=5).decision == "block"
    assert one(c, t, withdrawal_horizon_days=4).decision == "allow"


def test_withdrawal_month_end_clamped():
    c = [withdrawal(amount="3000", start=date(2026, 1, 31))]  # Feb occurrence lands on the 28th
    result = check(
        cset(*c), make_account(), [trade("buy", quantity="100")],
        as_of=date(2026, 2, 27), master=make_master(),
    )
    assert result.decisions[0].decision == "block"


# ---- capital gains budget ----------------------------------------------------------------

def test_gains_budget_boundary():
    c = [gains_budget(max_gain="1000")]
    assert one(c, trade("sell", "MSFT", "20", "200", ["M1"])).decision == "allow"  # exactly 1,000
    d = one(c, trade("sell", "MSFT", "21", "200", ["M1"]))
    assert d.decision == "block" and codes(d) == {"GAINS_BUDGET"}


def test_gains_budget_counts_ytd():
    d = one([gains_budget(max_gain="1000")], trade("sell", "MSFT", "20", "200", ["M1"]),
            account=make_account(realized_gains_ytd="100"))
    assert d.decision == "block"


def test_loss_sale_allowed_even_when_over_budget():
    acct = make_account(realized_gains_ytd="5000")
    c = [gains_budget(max_gain="1000")]
    assert one(c, trade("sell", "AAPL", "10", "100", ["L2"]), account=acct).decision == "allow"
    assert one(c, trade("sell", "MSFT", "1", "200", ["M1"]), account=acct).decision == "block"


def test_gains_budget_other_tax_year_inactive():
    d = one([gains_budget(tax_year=2025, max_gain="1")], trade("sell", "MSFT", "20", "200", ["M1"]))
    assert d.decision == "allow" and "INACTIVE_TAX_YEAR" in note_codes(d)


def test_gains_budget_term_filter():
    short_sale = trade("sell", "MSFT", "10", "200", ["M2"])  # short-term gain of 1,000
    assert one([gains_budget(max_gain="500", term="long")], short_sale).decision == "allow"
    assert one([gains_budget(max_gain="500", term="short")], short_sale).decision == "block"


def test_unspecified_lots_sold_fifo():
    t = trade("sell", "MSFT", "90", "200")  # 80 @ +50 (M1) then 10 @ +100 (M2) = 5,000
    assert one([gains_budget(max_gain="5000")], t).decision == "allow"
    assert one([gains_budget(max_gain="4999")], t).decision == "block"


# ---- lot holds ---------------------------------------------------------------------------

def test_lot_hold_is_lot_specific():
    c = [lot_hold(lot_ids=["L1"])]
    held = one(c, trade("sell", "AAPL", "10", lot_ids=["L1"]))
    assert held.decision == "block" and codes(held) == {"LOT_HOLD"}
    assert one(c, trade("sell", "AAPL", "10", lot_ids=["L2"])).decision == "allow"
    assert one(c, trade("buy", "AAPL")).decision == "allow"


def test_lot_hold_all_lots_when_none():
    assert one([lot_hold(lot_ids=None)], trade("sell", "AAPL", "1", lot_ids=["L2"])).decision == "block"
    assert one([lot_hold(lot_ids=None)], trade("sell", "AAPL", "1")).decision == "block"


def test_unspecified_sale_skips_held_lots_when_enough_remain():
    c = [lot_hold(lot_ids=["L1"])]
    assert one(c, trade("sell", "AAPL", "50")).decision == "allow"  # all from L2
    assert one(c, trade("sell", "AAPL", "60")).decision == "block"  # would need 10 from L1


def test_lot_hold_expiry():
    d = one([lot_hold(hold_until=date(2026, 10, 14))], trade("sell", "AAPL", "10", lot_ids=["L1"]))
    assert d.decision == "allow" and "EXPIRED" in note_codes(d)
    assert one([lot_hold(hold_until=AS_OF)], trade("sell", "AAPL", "10", lot_ids=["L1"])).decision == "block"


@pytest.mark.parametrize(
    "t",
    [trade("sell", "AAPL", "500"), trade("sell", "AAPL", "1", lot_ids=["ZZ"]), trade("sell", "XOM", "1")],
    ids=["oversell", "unknown_lot", "not_held"],
)
def test_invalid_sales_blocked(t):
    assert codes(one([], t)) == {"INVALID_TRADE"}


# ---- sector / ESG exclusions -------------------------------------------------------------

def test_theme_exclusion_blocks_buys_not_sells():
    c = [exclusion(theme="tobacco")]
    d = one(c, trade("buy", "MO"))
    assert d.decision == "block" and codes(d) == {"EXCLUDED"}
    assert one(c, trade("buy", "MSFT")).decision == "allow"


def test_sic_exclusion():
    c = [exclusion(basis="sic", sic_codes=["3571"])]
    assert one(c, trade("buy", "AAPL")).decision == "block"


def test_divest_exclusion_still_allows_selling():
    c = [exclusion(basis="sic", sic_codes=["3571"], applies_to="no_buy_and_divest")]
    assert one(c, trade("sell", "AAPL", "10", lot_ids=["L1"])).decision == "allow"


def test_unknown_sic_never_blocks():
    c = [exclusion(theme="tobacco")]
    assert note_codes(one(c, trade("buy", "UNKN"))) == {"UNCLASSIFIED"}
    assert one(c, trade("buy", "UNKN")).decision == "allow"
    assert one(c, trade("buy", "MO"), master=None).decision == "allow"


def test_custom_theme_table_overrides_default():
    c = [exclusion(theme="tobacco")]
    assert one(c, trade("buy", "MO"), theme_table={"tobacco": frozenset({"9999"})}).decision == "allow"


# ---- conflicts ---------------------------------------------------------------------------

def test_lot_hold_vs_divest_conflict():
    hold, ex = lot_hold("AAPL"), exclusion(basis="sic", sic_codes=["3571"], applies_to="no_buy_and_divest")
    found = find_conflicts(cset(hold, ex), make_master(), None, AS_OF)
    assert [(c.code, set(c.constraint_ids)) for c in found] == [("LOT_HOLD_VS_DIVEST", {hold.id, ex.id})]
    result = check(cset(hold, ex), make_account(), [], as_of=AS_OF, master=make_master())
    assert [c.code for c in result.conflicts] == ["LOT_HOLD_VS_DIVEST"]


def test_no_conflict_cases():
    divest = exclusion(basis="sic", sic_codes=["3571"], applies_to="no_buy_and_divest")
    plain = exclusion(basis="sic", sic_codes=["3571"], applies_to="no_buy")
    expired = lot_hold("AAPL", hold_until=date(2026, 1, 1))
    other = lot_hold("MSFT")
    assert find_conflicts(cset(lot_hold("AAPL"), plain), make_master(), None, AS_OF) == []
    assert find_conflicts(cset(expired, divest), make_master(), None, AS_OF) == []
    assert find_conflicts(cset(other, divest), make_master(), None, AS_OF) == []


# ---- clarifications, approval, misc ------------------------------------------------------

def test_clarification_blocks_related_security_only():
    c = [clarification(related="AAPL")]
    d = one(c, trade("buy", "AAPL"))
    assert d.decision == "block" and codes(d) == {"UNRESOLVED_CLARIFICATION"}
    assert one(c, trade("buy", "MSFT")).decision == "allow"


def test_clarification_without_security_warns():
    result = check(cset(clarification()), make_account(), [trade("buy")], as_of=AS_OF, master=make_master())
    assert result.decisions[0].decision == "allow" and result.warnings


def test_draft_set_rejected_by_default():
    with pytest.raises(NotApprovedError):
        check(cset(restriction(), approved=False), make_account(), [trade()], as_of=AS_OF)
    result = check(cset(restriction(), approved=False), make_account(), [trade()],
                   as_of=AS_OF, require_approved=False)
    assert result.decisions[0].decision == "block"


def test_empty_set_allows_everything():
    d = one([], trade("buy", "XOM"))
    assert d.decision == "allow" and d.reasons == []


def test_all_violations_reported():
    c = [restriction(action="no_buy"), cap(max_weight="0.1"), floor(min_amount="9999")]
    assert codes(one(c, trade("buy"))) == {"RESTRICTED", "CONCENTRATION_CAP", "CASH_FLOOR"}


def test_trades_evaluated_independently_in_input_order():
    c = [floor(min_amount="3000")]
    trades = [trade("buy", "MSFT", "30", "200"), trade("buy", "AAPL", "60", "100")]
    # each leaves 4,000 on its own; together they would breach, but there is no batch state
    result = run(c, trades)
    assert [d.decision for d in result] == ["allow", "allow"]
    assert [d.trade.ticker for d in result] == ["MSFT", "AAPL"]

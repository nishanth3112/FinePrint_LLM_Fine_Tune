import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from factories import (
    cap,
    clarification,
    cset,
    exclusion,
    floor,
    gains_budget,
    lot_hold,
    restriction,
    withdrawal,
)
from pydantic import ValidationError

from fineprint.schema import (
    CashFloor,
    ConstraintSet,
    SecurityRestriction,
    normalize_ticker,
    parse_constraint,
    validate_evidence,
)
from fineprint.themes import load_theme_table

ALL = [
    restriction(until=date(2026, 12, 31)),
    exclusion(),
    exclusion(basis="sic", sic_codes=["3571", "1311"]),
    cap(),
    floor(min_amount="5000"),
    gains_budget(),
    lot_hold(lot_ids=["L1"]),
    withdrawal(),
    clarification(related="AAPL"),
]


@pytest.mark.parametrize("c", ALL, ids=lambda c: c.type)
def test_round_trip(c):
    assert parse_constraint(json.loads(c.model_dump_json())) == c


def test_unknown_type_rejected():
    with pytest.raises(ValidationError):
        parse_constraint({"type": "bogus", "evidence": "x"})


def test_extra_field_rejected():
    with pytest.raises(ValidationError):
        parse_constraint(
            {"type": "cash_floor", "min_amount": "1", "evidence": "x", "surprise": 1}
        )


def test_id_optional_and_unique():
    a, b = restriction(), restriction()
    assert a.id and b.id and a.id != b.id


@pytest.mark.parametrize("ev", [None, ""])
def test_evidence_required(ev):
    payload = {"type": "cash_floor", "min_amount": "1"}
    if ev is not None:
        payload["evidence"] = ev
    with pytest.raises(ValidationError):
        parse_constraint(payload)


@pytest.mark.parametrize("kw", [{}, {"min_amount": "1", "min_weight": "0.1"}])
def test_cash_floor_exactly_one(kw):
    with pytest.raises(ValidationError):
        CashFloor(evidence="x", **kw)


@pytest.mark.parametrize("w", ["0", "-0.1", "1.01", "150%", 5])
def test_weight_bounds(w):
    with pytest.raises(ValidationError):
        cap(max_weight=w)


@pytest.mark.parametrize("w", ["5%", " 5 % ", 0.05, "0.05"])
def test_weight_coercion(w):
    assert cap(max_weight=w).max_weight == Decimal("0.05")


@pytest.mark.parametrize("raw", ["brk-b", " BRK/B ", "BRK.B", "brk.b"])
def test_ticker_normalization(raw):
    assert normalize_ticker(raw) == "BRK.B"
    assert restriction(ticker=raw).security.ticker == "BRK.B"


@pytest.mark.parametrize("raw", ["", "A B", "TOOLONGTICKER", "12$"])
def test_bad_ticker_rejected(raw):
    with pytest.raises(ValidationError):
        restriction(ticker=raw)


def test_sic_codes_sorted_deduped_and_four_digits():
    assert exclusion(basis="sic", sic_codes=["3571", "1311", "3571"]).sic_codes == ["1311", "3571"]
    with pytest.raises(ValidationError):
        exclusion(basis="sic", sic_codes=["35"])


def test_exclusion_basis_consistency():
    with pytest.raises(ValidationError):
        exclusion(basis="sic", sic_codes=[])
    with pytest.raises(ValidationError):
        exclusion(basis="theme", theme=None)
    with pytest.raises(ValidationError):
        exclusion(basis="theme", theme="tobacco", sic_codes=["2111"])
    with pytest.raises(ValidationError):
        exclusion(basis="theme", theme="not_a_theme")


def test_cap_target_rules():
    assert cap(target="brk-b").target == "BRK.B"
    assert cap(scope="sector", target="35").target == "35"
    with pytest.raises(ValidationError):
        cap(scope="single_security", target=None)
    with pytest.raises(ValidationError):
        cap(scope="sector", target="3571")


def test_lot_hold_lot_ids():
    assert lot_hold(lot_ids=["B", "A", "A"]).lot_ids == ["A", "B"]
    with pytest.raises(ValidationError):
        lot_hold(lot_ids=[])


def test_withdrawal_end_before_start_rejected():
    with pytest.raises(ValidationError):
        withdrawal(start=date(2026, 5, 1), end=date(2026, 4, 1))


def test_approval_invariant():
    assert ConstraintSet(constraints=[]).status == "draft"
    with pytest.raises(ValidationError):
        ConstraintSet(constraints=[], status="approved")
    assert cset(restriction()).status == "approved"


def test_cfa_categories():
    expected = {
        "security_restriction": "unique_circumstances",
        "sector_esg_exclusion": "unique_circumstances",
        "concentration_cap": "risk_objectives",
        "cash_floor": "liquidity",
        "capital_gains_budget": "tax_concerns",
        "lot_hold": "tax_concerns",
        "recurring_withdrawal": "liquidity",
        "clarification_needed": None,
    }
    assert {c.type: type(c).cfa_category for c in ALL} == expected


def test_validate_evidence_flags_non_substrings():
    text = "Please never sell my Apple shares. Keep 5% in cash."
    good = SecurityRestriction(
        action="no_sell", security={"ticker": "AAPL"}, evidence="never sell my Apple shares"
    )
    wrong_case = SecurityRestriction(
        action="no_sell", security={"ticker": "AAPL"}, evidence="NEVER SELL"
    )
    invented = CashFloor(min_weight="0.05", evidence="keep ten percent in cash")
    flags = validate_evidence(text, cset(good, wrong_case, invented))
    assert [f.index for f in flags] == [1, 2]
    assert flags[0].constraint_id == wrong_case.id
    assert flags[1].evidence == "keep ten percent in cash"
    assert validate_evidence(text, cset(good)) == []


def test_theme_table_is_versioned_and_sourced():
    table = load_theme_table()
    assert set(table) == {"tobacco", "weapons", "fossil_fuels", "alcohol", "gambling"}
    assert all(c.isdigit() and len(c) == 4 for codes in table.values() for c in codes)
    raw = json.loads(Path("configs/theme_sic.json").read_text())
    assert raw["version"] >= 1 and raw["source_url"].startswith("https://www.sec.gov/")
    assert "approximation" in raw["note"].lower()

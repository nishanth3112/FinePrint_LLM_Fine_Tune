import json
from datetime import date

import pytest
from factories import (
    cap,
    clarification,
    exclusion,
    floor,
    gains_budget,
    lot_hold,
    restriction,
    withdrawal,
)

from fineprint.eval.match import match_constraints, normalize, score

N_BOOT = 2000


def test_identical_sets_perfect():
    gold = [restriction("AAPL"), cap(max_weight="0.1"), floor(min_amount="5000")]
    pred = [restriction("AAPL"), cap(max_weight="0.1"), floor(min_amount="5000")]
    report = score([(gold, pred)], n_boot=N_BOOT, seed=0)
    assert (report.strict.precision, report.strict.recall, report.strict.f1) == (1.0, 1.0, 1.0)
    assert report.counts.exact == 3
    assert report.value_errors == report.hallucinated == report.missed == []


def test_normalization_variants_match():
    gold = [restriction("BRK.B"), cap(max_weight="0.05"), exclusion("sic", sic_codes=["3571", "1311"])]
    pred = [
        {"type": "security_restriction", "action": "no_trade", "security": {"ticker": "brk-b"}, "evidence": "e"},
        {"type": "concentration_cap", "scope": "single_security", "target": "AAPL",
         "max_weight": "5%", "evidence": "e"},
        {"type": "sector_esg_exclusion", "basis": "sic", "sic_codes": ["1311", "3571"],
         "applies_to": "no_buy", "evidence": "e"},
    ]
    assert match_constraints(gold, pred).hallucinated == []
    assert score([(gold, pred)], n_boot=N_BOOT).strict.f1 == 1.0


def test_ids_evidence_and_free_text_not_scored():
    gold = [restriction("AAPL", id="g1", evidence="never sell Apple"),
            lot_hold("AAPL", reason="inherited"), clarification("AAPL")]
    pred = [restriction("AAPL", id="p9", evidence="completely different quote"),
            lot_hold("AAPL", reason="something else"),
            clarification("AAPL", candidates=["x"]).model_copy(update={"question": "Another question?"})]
    assert score([(gold, pred)], n_boot=N_BOOT).strict.f1 == 1.0


def test_value_error_is_not_hallucinated_plus_missed():
    gold = [restriction("AAPL", action="no_sell"), cap(max_weight="0.1")]
    pred = [restriction("AAPL", action="no_buy"), cap(max_weight="0.1")]
    report = score([(gold, pred)], n_boot=N_BOOT)
    assert report.counts.value_error == 1 and report.counts.hallucinated == 0 and report.counts.missed == 0
    err = report.value_errors[0]
    assert (err.type, err.field, err.gold, err.pred) == ("security_restriction", "action", "no_sell", "no_buy")
    # strict punishes the value error; key-level credits the right identification
    assert report.strict.precision == 0.5 and report.strict.recall == 0.5
    assert report.key_level.f1 == 1.0


def test_hallucinated_and_missed():
    gold = [restriction("AAPL"), floor(min_amount="5000")]
    pred = [restriction("AAPL"), restriction("TSLA")]
    m = match_constraints(gold, pred)
    assert [c.type for c in m.missed] == ["cash_floor"]
    assert [c.security.ticker for c in m.hallucinated] == ["TSLA"]
    report = score([(gold, pred)], n_boot=N_BOOT)
    assert report.hallucinated[0].key.endswith("TSLA") and report.missed[0].type == "cash_floor"
    assert report.strict.precision == 0.5 and report.strict.recall == 0.5


def test_duplicates_are_multisets():
    gold = [restriction("AAPL", action="no_buy"), restriction("AAPL", action="no_sell")]
    assert match_constraints(gold, list(reversed(gold))).hallucinated == []
    m = match_constraints(gold, [restriction("AAPL", action="no_buy")])
    assert len(m.exact) == 1 and len(m.missed) == 1


def test_exact_pairs_preferred_over_same_key_value_errors():
    gold = [restriction("AAPL", action="no_buy"), restriction("AAPL", action="no_sell")]
    pred = [restriction("AAPL", action="no_sell"), restriction("AAPL", action="no_trade")]
    m = match_constraints(gold, pred)
    assert len(m.exact) == 1 and len(m.value_errors) == 1 and not m.hallucinated and not m.missed


def test_clarification_keyed_on_related_security_only():
    assert match_constraints([clarification("AAPL")], [clarification("AAPL", kind="other")]).exact
    m = match_constraints([clarification("AAPL")], [clarification("MSFT")])
    assert len(m.hallucinated) == 1 and len(m.missed) == 1
    assert match_constraints([clarification()], [clarification()]).exact


def test_every_type_has_a_key_and_values():
    for c in [restriction(), exclusion(), cap(), floor(min_amount="1"), gains_budget(),
              lot_hold(), withdrawal(), clarification()]:
        n = normalize(c)
        assert n.type == c.type and isinstance(n.key, str)


def test_empty_edge_cases():
    both = score([([], [])], n_boot=N_BOOT)
    assert both.strict.f1 == 1.0
    no_pred = score([([restriction()], [])], n_boot=N_BOOT)
    assert (no_pred.strict.precision, no_pred.strict.recall, no_pred.strict.f1) == (0.0, 0.0, 0.0)
    no_gold = score([([], [restriction()])], n_boot=N_BOOT)
    assert (no_gold.strict.precision, no_gold.strict.recall, no_gold.strict.f1) == (0.0, 0.0, 0.0)
    nothing = score([], n_boot=N_BOOT)
    assert nothing.n_examples == 0 and nothing.ci is None


def test_per_type_breakdown():
    gold = [restriction("AAPL"), cap(max_weight="0.1")]
    pred = [restriction("AAPL"), cap(max_weight="0.2")]
    report = score([(gold, pred)], n_boot=N_BOOT)
    assert report.by_type["security_restriction"].strict.f1 == 1.0
    assert report.by_type["concentration_cap"].strict.f1 == 0.0
    assert report.by_type["concentration_cap"].key_level.f1 == 1.0


def _examples():
    # Varied sizes and error mixes so the bootstrap distribution is not a handful of values.
    out = []
    for i in range(30):
        n = 1 + i % 5
        gold = [restriction("A" + chr(65 + j)) for j in range(n)]
        pred = gold[: n - (i % 3 == 0)] + ([cap(max_weight="0.2")] if i % 4 == 0 else [])
        out.append((gold, pred))
    return out


def test_bootstrap_deterministic_and_brackets_point_estimate():
    a = score(_examples(), n_boot=N_BOOT, seed=7)
    b = score(_examples(), n_boot=N_BOOT, seed=7)
    c = score(_examples(), n_boot=N_BOOT, seed=8)
    assert a.ci == b.ci and a.ci != c.ci
    lo, hi = a.ci["strict_f1"]
    assert lo <= a.strict.f1 <= hi
    assert set(a.ci) == {f"{k}_{m}" for k in ("strict", "key") for m in ("precision", "recall", "f1")}


def test_bootstrap_resamples_examples_not_constraints():
    perfect = ([restriction("AAPL"), cap()], [restriction("AAPL"), cap()])
    miss = ([restriction("MSFT"), cap(target="MSFT")], [])
    report = score([perfect, miss], n_boot=N_BOOT, seed=0)
    lo, hi = report.ci["strict_f1"]
    assert lo == 0.0 and hi == 1.0  # drawing the same example twice reaches both extremes


def test_report_is_json_serializable():
    report = score(_examples(), n_boot=200, seed=0)
    json.loads(report.model_dump_json())
    assert report.n_boot == 200 and report.seed == 0 and report.n_examples == 30


def test_dates_compare_after_normalization():
    gold = [restriction("AAPL", until=date(2026, 12, 31))]
    pred = [{"type": "security_restriction", "action": "no_trade", "security": {"ticker": "AAPL"},
             "until": "2026-12-31", "evidence": "e"}]
    assert score([(gold, pred)], n_boot=N_BOOT).strict.f1 == 1.0
    pred[0]["until"] = "2027-01-01"
    assert score([(gold, pred)], n_boot=N_BOOT).value_errors[0].field == "until"


@pytest.mark.parametrize("n_boot", [0, -1])
def test_bad_n_boot_rejected(n_boot):
    with pytest.raises(ValueError):
        score([([], [])], n_boot=n_boot)

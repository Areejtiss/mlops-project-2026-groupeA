"""Tests for the fairness audit (D5, D8: only ACCEPTE counts as accepted)."""

import numpy as np
import pandas as pd
import pytest

from crediscore.fairness.audit import audit, d5_checks, gap_pts
from crediscore.models.cost import ACCEPT, REFUSE, REVIEW

A, V, R = ACCEPT, REVIEW, REFUSE


def frame(gender: list[object], age: list[float], target: list[int]) -> pd.DataFrame:
    return pd.DataFrame({"CODE_GENDER": gender, "AGE_YEARS": age, "TARGET": target})


def test_hand_made_frame_with_known_gaps() -> None:
    # F: 4 rows, accepted 3 (raw 0.75); good payers 3, accepted 2 (EO 2/3);
    #    1 default, accepted (recall 0.0).
    # M: 4 rows, accepted 0 (raw 0.0); good payers 2, accepted 0 (EO 0);
    #    2 defaults, both in review (recall 1.0).
    # The three gaps differ (75, 66.7, 100): none can stand in for another.
    df = frame(
        ["F", "F", "F", "F", "M", "M", "M", "M"],
        [25.0, 35.0, 55.0, 45.0, 25.0, 35.0, 55.0, 29.0],
        [0, 0, 0, 1, 0, 0, 1, 1],
    )
    d = np.array([A, A, R, A, R, V, V, V])
    g = audit(df, d, target="TARGET")["gender"]
    assert g["acceptance_by_group"] == pytest.approx({"F": 0.75, "M": 0.0})
    assert g["raw_gap_pts"] == pytest.approx(75.0)
    assert g["acceptance_good_payers_by_group"] == pytest.approx({"F": 2 / 3, "M": 0})
    assert g["eo_gap_pts"] == pytest.approx(100 * 2 / 3)
    assert g["recall_by_group"] == pytest.approx({"F": 0.0, "M": 1.0})
    assert g["recall_gap_pts"] == pytest.approx(100.0)
    assert g["excluded_rows"] == 0


def test_age_bands_are_left_closed() -> None:
    df = frame(["F"] * 6, [29.99, 30.0, 49.99, 50.0, 18.0, 80.0], [0] * 6)
    d = np.array([A, R, R, A, A, A])
    a = audit(df, d, target="TARGET")["age"]
    # <30: 29.99, 18 -> 1.0; 30-50: 30, 49.99 -> 0.0; >=50: 50, 80 -> 1.0.
    assert a["acceptance_by_group"] == pytest.approx(
        {"<30": 1.0, "30-50": 0.0, "≥50": 1.0}
    )
    assert a["raw_gap_pts"] == pytest.approx(100.0)
    assert a["eo_gap_pts"] == pytest.approx(100.0)


def test_review_and_refuse_are_not_accepted_but_both_detect_defaults() -> None:
    df = frame(["F", "F", "M", "M"], [40.0] * 4, [1, 1, 1, 1])
    g = audit(df, np.array([V, R, A, A]), target="TARGET")["gender"]
    assert g["acceptance_by_group"] == pytest.approx({"F": 0.0, "M": 1.0})
    assert g["recall_by_group"] == pytest.approx({"F": 1.0, "M": 0.0})


def test_xna_and_missing_gender_are_excluded_and_counted() -> None:
    df = frame(["F", "M", "XNA", None], [40.0] * 4, [0] * 4)
    g = audit(df, np.array([A, R, A, A]), target="TARGET")["gender"]
    assert set(g["acceptance_by_group"]) == {"F", "M"}
    assert g["excluded_rows"] == 2
    assert g["raw_gap_pts"] == pytest.approx(100.0)


def test_missing_age_is_excluded_and_counted() -> None:
    df = frame(["F", "F", "F"], [25.0, np.nan, 60.0], [0, 0, 0])
    a = audit(df, np.array([A, A, R]), target="TARGET")["age"]
    assert a["excluded_rows"] == 1
    assert a["acceptance_by_group"] == pytest.approx({"<30": 1.0, "≥50": 0.0})


def test_single_group_has_no_gap() -> None:
    df = frame(["F", "F"], [40.0, 41.0], [0, 1])
    out = audit(df, np.array([A, R]), target="TARGET")
    for side in ("gender", "age"):
        assert out[side]["raw_gap_pts"] == 0.0
        assert out[side]["eo_gap_pts"] == 0.0
        assert out[side]["recall_gap_pts"] == 0.0


def test_empty_band_and_group_without_good_payers_are_left_out() -> None:
    # No row >=50; the 30-50 band has only defaults (no EO rate).
    df = frame(["F", "M", "F"], [25.0, 35.0, 40.0], [0, 1, 1])
    a = audit(df, np.array([A, R, A]), target="TARGET")["age"]
    assert set(a["acceptance_by_group"]) == {"<30", "30-50"}
    assert set(a["acceptance_good_payers_by_group"]) == {"<30"}
    assert a["eo_gap_pts"] == 0.0
    assert set(a["recall_by_group"]) == {"30-50"}


def test_every_row_excluded_gives_empty_groups() -> None:
    df = frame(["XNA", "XNA"], [np.nan, np.nan], [0, 1])
    out = audit(df, np.array([A, R]), target="TARGET")
    for side in ("gender", "age"):
        assert out[side]["acceptance_by_group"] == {}
        assert out[side]["raw_gap_pts"] == 0.0
        assert out[side]["excluded_rows"] == 2


def test_audit_follows_the_frame_index() -> None:
    df = frame(["F", "M"], [25.0, 60.0], [0, 0]).set_axis([10, 3])
    out = audit(df, np.array([A, R]), target="TARGET")
    assert out["gender"]["acceptance_by_group"] == pytest.approx({"F": 1.0, "M": 0.0})


def test_gap_pts() -> None:
    assert gap_pts(pd.Series([0.2, 0.7, 0.5])) == pytest.approx(50.0)
    assert gap_pts(pd.Series([0.4])) == 0.0
    assert gap_pts(pd.Series([], dtype=float)) == 0.0


# --- D5 checks ---------------------------------------------------------------


def fair(gender_raw: float, gender_eo: float, age_eo: float) -> dict:
    return {
        "gender": {"raw_gap_pts": gender_raw, "eo_gap_pts": gender_eo},
        "age": {"raw_gap_pts": 50.0, "eo_gap_pts": age_eo},
    }


def checks(f: dict, anchor: float = 38.7) -> dict:
    return d5_checks(
        f, age_anchor_pts=anchor, max_gender_gap_pts=10.0, age_gap_margin_pts=2.0
    )


def test_d5_checks_values_limits_and_inclusive_bounds() -> None:
    out = checks(fair(10.0, 10.0, 40.7), anchor=38.7)
    assert out == {
        "gender_raw_gap": {"value": 10.0, "limit": 10.0, "pass": True},
        "gender_eo_gap": {"value": 10.0, "limit": 10.0, "pass": True},
        "age_eo_gap": {"value": 40.7, "limit": pytest.approx(40.7), "pass": True},
    }


@pytest.mark.parametrize(
    ("f", "failing"),
    [
        (fair(10.001, 5.0, 30.0), "gender_raw_gap"),
        (fair(5.0, 10.001, 30.0), "gender_eo_gap"),
        (fair(5.0, 5.0, 40.701), "age_eo_gap"),
    ],
)
def test_d5_checks_each_condition_fails_alone(f: dict, failing: str) -> None:
    out = checks(f)
    assert [k for k, c in out.items() if not c["pass"]] == [failing]


def test_d5_checks_ignore_float_noise_on_the_bound() -> None:
    # 100 * (0.4 - 0.3) is 10.000000000000004 in floats: still a 10-pt gap.
    noisy = 100 * (0.4 - 0.3)
    assert noisy > 10.0
    assert checks(fair(noisy, noisy, 30.0))["gender_raw_gap"]["pass"] is True


def test_d5_checks_use_the_given_tolerances() -> None:
    out = d5_checks(
        fair(6.0, 6.0, 33.0),
        age_anchor_pts=30.0,
        max_gender_gap_pts=5.0,
        age_gap_margin_pts=3.0,
    )
    assert out["gender_raw_gap"]["limit"] == 5.0
    assert out["gender_raw_gap"]["pass"] is False
    assert out["age_eo_gap"] == {"value": 33.0, "limit": 33.0, "pass": True}

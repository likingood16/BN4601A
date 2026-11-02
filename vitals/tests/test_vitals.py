"""Checks for the vitals pipeline. Run:  python tests/test_vitals.py   (from vitals/)
Also works with pytest if it is installed."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd

import ktas_data as kd
import news2
from vitals_model import VitalsModel, enforce_floor

VM = VitalsModel()


def test_news2_band_edges():
    # Every boundary of the RCP NEWS2 chart (2017), SpO2 scale 1
    cases = [("rr", 8, 3), ("rr", 9, 1), ("rr", 11, 1), ("rr", 12, 0), ("rr", 20, 0), ("rr", 21, 2),
             ("rr", 24, 2), ("rr", 25, 3), ("spo2", 91, 3), ("spo2", 92, 2), ("spo2", 93, 2),
             ("spo2", 94, 1), ("spo2", 95, 1), ("spo2", 96, 0), ("sbp", 90, 3), ("sbp", 91, 2),
             ("sbp", 100, 2), ("sbp", 101, 1), ("sbp", 110, 1), ("sbp", 111, 0), ("sbp", 219, 0),
             ("sbp", 220, 3), ("hr", 40, 3), ("hr", 41, 1), ("hr", 50, 1), ("hr", 51, 0), ("hr", 90, 0),
             ("hr", 91, 1), ("hr", 110, 1), ("hr", 111, 2), ("hr", 130, 2), ("hr", 131, 3),
             ("temp_c", 35.0, 3), ("temp_c", 35.1, 1), ("temp_c", 36.0, 1), ("temp_c", 36.1, 0),
             ("temp_c", 38.0, 0), ("temp_c", 38.1, 1), ("temp_c", 39.0, 1), ("temp_c", 39.1, 2)]
    for param, value, expected in cases:
        assert news2.band_points(param, value) == expected, (param, value)


def test_risk_bands():
    assert news2.score(rr=16, spo2=98, sbp=120, hr=80, temp_c=37, consciousness="alert")["risk"] == "low"
    assert news2.score(rr=16, spo2=98, sbp=120, hr=80, temp_c=37, consciousness="pain")["risk"] == "low-medium"
    # RR 22 (+2) + SpO2 94 (+1) + SBP 100 (+2) = 5
    assert news2.score(rr=22, spo2=94, sbp=100, hr=80, temp_c=37, consciousness="alert")["risk"] == "medium"
    assert news2.score(rr=22, spo2=89, sbp=118, hr=124, temp_c=37, consciousness="alert")["risk"] == "high"
    assert news2.score(rr=16, spo2=98, sbp=120, hr=80, temp_c=37, consciousness="alert",
                       on_oxygen=True)["news2"] == 2


def test_cardiac_arrest_is_never_low_risk():
    out = VM.assess(consciousness="unresponsive")
    assert out["news2_risk"] == "high" and out["danger"] and out["floor_priority"] <= 2
    out = VM.assess(consciousness="alert")          # no vitals at all
    assert out["news2_risk"] == "high" and out["escalated"]


def test_floor_only_raises_urgency():
    for llm in range(1, 6):
        for floor in range(1, 6):
            assert enforce_floor(llm, floor) == min(llm, floor)
    assert enforce_floor(5, 2) == 2 and enforce_floor(1, 3) == 1


def test_input_cleaning():
    out = VM.assess(rr=18, spo2=98, sbp=130, hr=80, temp_c=101.3, consciousness="alert")
    assert any("°F" in w for w in out["warnings"])
    assert out["subscores"]["temp_c"] == 1                       # 38.5 °C -> +1
    out = VM.assess(rr=18, spo2=9, sbp=130, hr=80, temp_c=37, consciousness="alert")
    assert "spo2" in out["missing"] and out["warnings"]
    out = VM.assess(rr=18, spo2=97, sbp=80, dbp=120, hr=80, temp_c=37, consciousness="alert")
    assert {"sbp"} <= set(out["missing"])
    out = VM.assess(rr=18, consciousness="sleepy")
    assert "consciousness" in out["missing"] and out["warnings"]


def test_output_is_json_and_ordered():
    sick = VM.assess(rr=26, spo2=88, sbp=88, hr=132, temp_c=39.4, consciousness="verbal", age=80)
    well = VM.assess(rr=16, spo2=99, sbp=122, hr=72, temp_c=36.8, consciousness="alert", age=25)
    json.dumps(sick), json.dumps(well)
    assert sick["p_urgent"] > well["p_urgent"]
    assert sick["floor_priority"] < well["floor_priority"]
    assert 0 <= well["p_urgent"] <= 1


def test_whole_dataset_matches_saved_scores():
    df = kd.load_clean()
    saved = pd.read_csv(kd.PROCESSED_DIR / "news2_scores.csv")
    rescored = news2.score_frame(df)
    assert (rescored["news2"].to_numpy() == saved.set_index("visit_id").loc[df.visit_id, "news2"].to_numpy()).all()


def test_clean_data_validates():
    df, _ = kd.clean(kd.load_raw())
    report = kd.validate(df)
    assert report[(report.severity == "error")].passed.all()
    assert len(df) == 1262


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, f in tests:
        f()
        print(f"PASS  {name}")
    print(f"\nall {len(tests)} tests passed")

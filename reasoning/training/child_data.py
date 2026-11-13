"""Children's patients (under 16) from NHAMCS 2022, for a children's urgency score (13 Nov 2026, feedback A2).

    python child_data.py            # -> processed/pool1_children.parquet + processed/cards_children.parquet

Why: KTAS starts at 15 and Caleb's NHAMCS table keeps only ages 16+ (vitals/src/nhamcs.py), so the urgency head never
saw a child. NHAMCS 2022 has 1,795 ED visits under 16 with a triage level (170 urgent, level 1-2).
Same coding as the adults: Caleb's harmonize() is reused unchanged (vitals/ is not edited): the children's ages are
shifted by +100 so his "16 or over" filter keeps them, then shifted back. Complaint text = the reasons for visit (RFV1-3),
exactly as notebook 02 did for the adults. Split 70/15/15, same urgent share in each, seed 42.
Cards are made exactly as the app makes them (cards.make_card + Caleb's vitals model), so the head sees what it will see
in the app.
"""
import sys as _sys; from pathlib import Path as _Path  # noqa: E401,E702  (reasoning/src etc. importable)
_sys.path[:0] = [str(_Path(__file__).resolve().parents[1] / _d) for _d in ("src", "training", "checks")]
import pandas as pd
from sklearn.model_selection import train_test_split

from cards import assess, dev_vitals_model, make_card
from config import PROCESSED, REPO, SEED

_sys.path.insert(0, str(REPO / "vitals" / "src"))
import nhamcs  # noqa: E402  (Caleb's file, read-only)

SHIFT = 100


def load_children():
    raw = nhamcs.load()
    kids = raw[(raw.AGE < 16)].copy()
    kids["AGE"] = kids["AGE"] + SHIFT
    t, _ = nhamcs.harmonize(kids)
    t["age"] = t["age"] - SHIFT

    with pd.io.stata.StataReader(nhamcs.RAW) as reader:
        code_to_text = reader.value_labels()["RFVF"]
    rfv = pd.read_stata(nhamcs.RAW, columns=["AGE", "IMMEDR", "RFV1", "RFV2", "RFV3"], convert_categoricals=False)
    rfv = rfv[(rfv.AGE < 16) & rfv.IMMEDR.between(1, 5)].reset_index(drop=True)      # same rows as harmonize()
    assert len(rfv) == len(t) and (rfv.AGE.values == t.age.values).all()

    def complaint(row):
        reasons = [code_to_text.get(c) for c in (row.RFV1, row.RFV2, row.RFV3) if c > 0]
        return "; ".join(r for r in reasons if r) or None

    t["chief_complaint"] = rfv.apply(complaint, axis=1)
    train, rest = train_test_split(t.index, test_size=0.30, stratify=t.urgent, random_state=SEED)
    val, test = train_test_split(rest, test_size=0.50, stratify=t.urgent[rest], random_state=SEED)
    t["split"] = "train"
    t.loc[val, "split"] = "val"
    t.loc[test, "split"] = "test"
    return pd.DataFrame({
        "source": "nhamcs_child", "id": "child_" + t.index.astype(str), "split": t["split"],
        "age": t["age"], "sex": t["is_male"].map({1.0: "male", 0.0: "female"}),
        "ambulance": t["ambulance"], "injured": t["injured"], "mental": None,
        "chief_complaint": t["chief_complaint"], "nrs_pain": t["nrs_pain"],
        "sbp": t["sbp"], "dbp": t["dbp"], "hr": t["hr"], "rr": t["rr"], "temp_c": t["temp_c"], "spo2": t["spo2"],
        "triage_level": t["triage_level"], "urgent": t["urgent"],
    })


def main():
    df = load_children()
    df.to_parquet(PROCESSED / "pool1_children.parquet", index=False)
    vm = dev_vitals_model()
    rows = []
    for _, r in df.iterrows():
        out = assess(vm, r)
        rows.append({"id": r["id"], "source": r["source"], "split": r["split"], "triage_level": r["triage_level"],
                     "urgent": r["urgent"], "card": make_card(r, out), "news2": out["news2"],
                     "vitals_p_urgent": out["p_urgent"], "floor_priority": out["floor_priority"],
                     "danger": out["danger"]})
    cards = pd.DataFrame(rows)
    cards.to_parquet(PROCESSED / "cards_children.parquet", index=False)
    print(pd.crosstab(cards.split, cards.urgent, margins=True))
    print(cards.card.iloc[0])


if __name__ == "__main__":
    main()

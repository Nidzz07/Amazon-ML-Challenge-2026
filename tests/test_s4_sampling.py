"""s4_train primitives: per-entity hard negatives, entity-safe streaming, entity-hash splits, and the calibration
design (isotonic on an UNSAMPLED slice recovers the real prevalence; on a 2:1 sample it reads high)."""
import numpy as np
import polars as pl
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression

import s4_train as s4


def _frame(rows):
    return pl.DataFrame(rows, schema={"source1_entity_id": pl.String, "label": pl.UInt8, "prior": pl.Float32})


def test_hard_negatives_are_top_per_entity_with_floor():
    rows = (
        [("A", 1, 0.1), ("A", 1, 0.2)] + [("A", 0, s) for s in (0.9, 0.8, 0.7, 0.6, 0.5)]   # 2 pos -> quota 4
        + [("B", 0, s) for s in (0.4, 0.3, 0.2, 0.1, 0.05)]                                    # 0 pos -> floor 3
        + [("C", 1, 0.5), ("C", 0, 0.05)]                                                       # only 1 neg exists
    )
    out = s4.select_hard_negatives(_frame(rows), "prior", ratio=2.0, min_negs=3)
    kept = {(e, l, round(p, 2)) for e, l, p in zip(out["source1_entity_id"], out["label"], out["prior"])}
    assert {("A", 0, 0.9), ("A", 0, 0.8), ("A", 0, 0.7), ("A", 0, 0.6)} <= kept
    assert ("A", 0, 0.5) not in kept
    assert sum(1 for k in kept if k[0] == "A" and k[1] == 1) == 2       # every positive kept, even a low prior one
    assert {k for k in kept if k[0] == "B"} == {("B", 0, 0.4), ("B", 0, 0.3), ("B", 0, 0.2)}
    assert {k for k in kept if k[0] == "C"} == {("C", 1, 0.5), ("C", 0, 0.05)}


def test_streaming_never_splits_an_entity(tmp_path):
    sizes = [1, 7, 3, 30, 2, 2, 11, 5, 1, 19]
    ids = [f"e{i}" for i, n in enumerate(sizes) for _ in range(n)]
    df = pl.DataFrame({"source1_entity_id": ids, "label": [0] * len(ids), "f000": np.arange(len(ids), dtype=np.float32)})
    path = tmp_path / "f.parquet"
    pq.write_table(df.to_arrow(), path, row_group_size=4)
    seen = []
    for batch in s4.iter_entity_batches(path, ["source1_entity_id", "label", "f000"], batch_rows=6):
        seen.append(set(batch["source1_entity_id"].unique()))
    assert sum(len(s) for s in seen) == len(sizes)                      # every entity in exactly one batch
    assert set().union(*seen) == set(dict.fromkeys(ids))
    total = sum(b.height for b in s4.iter_entity_batches(path, ["source1_entity_id", "label", "f000"], batch_rows=6))
    assert total == len(ids)


def test_roles_are_disjoint_deterministic_and_proportioned():
    ids = pl.Series([f"ent{i}" for i in range(20000)])
    role = s4.assign_role(s4.entity_bucket(ids), val_frac=0.2, calib_frac=0.1)
    assert np.array_equal(role, s4.assign_role(s4.entity_bucket(ids), 0.2, 0.1))
    frac = [np.mean(role == r) for r in (s4.TRAIN, s4.VAL, s4.CALIB)]
    assert abs(frac[0] - 0.7) < 0.02 and abs(frac[1] - 0.2) < 0.02 and abs(frac[2] - 0.1) < 0.02
    # a row's role depends only on its entity id, so batch boundaries cannot move an entity between splits
    assert np.array_equal(role[:100], s4.assign_role(s4.entity_bucket(ids[:100]), 0.2, 0.1))


def test_isotonic_on_unsampled_slice_recovers_real_prevalence():
    rng = np.random.default_rng(0)
    n = 200_000
    y = (rng.random(n) < 0.115).astype(np.uint8)                        # ~7.7 neg per pos, as at inference
    score = np.clip(0.5 * y + rng.normal(0.25, 0.15, n), 0, 1)          # informative but imperfect raw score
    bucket = rng.integers(0, 100, n).astype(np.uint8)
    ir, stats = s4.fit_isotonic_with_check(score, y, bucket)
    assert abs(ir.transform(score).mean() - y.mean()) < 0.005          # calibration in the large
    assert stats["ece_cv"] < 0.01

    # the OLD design: fit isotonic on a 2:1 resample, apply at natural prevalence -> probabilities run high
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    keep = np.concatenate([pos, rng.choice(neg, 2 * len(pos), replace=False)])
    old = IsotonicRegression(out_of_bounds="clip").fit(score[keep], y[keep])
    old_err = old.transform(score).mean() - y.mean()
    new_err = abs(ir.transform(score).mean() - y.mean())
    assert old_err > 0.02                                               # reads high (~+30% here; worse when the score is less separable)
    assert new_err < old_err / 5


def test_reliability_table_bins_and_calibrated_data_reads_straight():
    rng = np.random.default_rng(3)
    n = 300_000
    p_true = rng.uniform(0, 1, n)
    y = (rng.random(n) < p_true).astype(np.uint8)                       # perfectly calibrated by construction
    rows = s4.reliability_table(y, p_true)
    assert sum(r["n"] for r in rows) == n and len(rows) == 10
    assert all(abs(r["mean_pred"] - r["observed"]) < 0.01 for r in rows)
    near = s4.reliability_table(y, p_true, [0.65, 0.75])
    assert len(near) == 1 and 0.65 <= near[0]["mean_pred"] <= 0.75 and abs(near[0]["mean_pred"] - near[0]["observed"]) < 0.01


def test_fit_isotonic_reports_cross_fitted_reliability():
    rng = np.random.default_rng(4)
    n = 120_000
    y = (rng.random(n) < 0.115).astype(np.uint8)
    score = np.clip(0.5 * y + rng.normal(0.25, 0.15, n), 0, 1)
    bucket = rng.integers(0, 100, n).astype(np.uint8)
    _, stats = s4.fit_isotonic_with_check(score, y, bucket)
    assert sum(r["n"] for r in stats["reliability"]) == n               # every row scored by a calibrator that never saw it
    hi = [r for r in stats["reliability"] if r["mean_pred"] > 0.6]
    assert hi and all(abs(r["mean_pred"] - r["observed"]) < 0.1 for r in hi)
    assert stats["near_0_7"] is None or 0.65 <= stats["near_0_7"]["mean_pred"] <= 0.75

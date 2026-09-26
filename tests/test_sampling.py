"""pipeline_io.sample_train_entities: the one shared train-entity sampler."""
import polars as pl
import pytest

import pipeline_io as pio


def s1_frame(counts: dict[str, int]) -> pl.DataFrame:
    rows = [(f"S1-{c}-{i:05d}", c) for c, n in counts.items() for i in range(n)]
    return pl.DataFrame(rows, schema={"entity_id": pl.String, "country": pl.String}, orient="row")


def per_country(s1: pl.DataFrame, ids: pl.Series) -> dict[str, int]:
    return dict(s1.filter(pl.col("entity_id").is_in(ids.implode())).group_by("country").len().iter_rows())


def test_test_split_is_never_sampled():
    with pytest.raises(AssertionError, match="test must never be sampled"):
        pio.sample_train_entities(s1_frame({"US": 10}), 5, "test", True)


def test_noop_returns_every_id():
    s1 = s1_frame({"India": 40, "US": 60})
    for n in (0, -1, 100, 800_000):
        assert pio.sample_train_entities(s1, n, "train", True).to_list() == s1["entity_id"].sort().to_list()


def test_stratified_exact_total_and_seeded():
    s1 = s1_frame({"India": 883, "US": 1_324})
    ids = pio.sample_train_entities(s1, 800, "train", True)
    assert ids.len() == 800 and ids.n_unique() == 800
    got = per_country(s1, ids)
    assert got == {"India": 320, "US": 480}  # 800 x 883/2207 = 320.07, 800 x 1324/2207 = 479.93
    # Depends only on the id set and seed, never on row order.
    assert pio.sample_train_entities(s1.reverse(), 800, "train", True).equals(ids)
    assert not pio.sample_train_entities(s1, 800, "train", True, seed=7).equals(ids)


def test_large_n_does_not_overflow():
    # Country counts are UInt32; count x n above 2**32 once overflowed and shrank the quotas.
    s1 = s1_frame({"India": 30_000, "US": 70_000})  # 70,000 x 99,999 > 2**32
    ids = pio.sample_train_entities(s1, 99_999, "train", True)
    assert ids.len() == 99_999 and per_country(s1, ids) == {"India": 30_000, "US": 69_999}
    ids = pio.sample_train_entities(s1.head(0), 800_000, "train", True)  # empty frame: no-op, no crash
    assert ids.len() == 0


def test_held_out_always_kept_and_not_counted(monkeypatch):
    s1 = s1_frame({"India": 400, "US": 600})
    keep = pl.Series(["S1-India-00000", "S1-US-00001", "S1-US-00002"])
    monkeypatch.setattr(pio, "val_ids", lambda smoke: keep)
    ids = pio.sample_train_entities(s1, 100, "train", False)
    assert ids.len() == 103 and set(keep) <= set(ids)
    rest = ids.filter(~ids.is_in(keep.implode()))
    assert per_country(s1, rest) == {"India": 40, "US": 60}


def test_full_run_without_val_split_refuses(monkeypatch):
    # A full run must never sample without the held-out ids: that would be a different set.
    monkeypatch.setattr(pio, "val_ids", lambda smoke: None)
    with pytest.raises(AssertionError, match="validation_split.py"):
        pio.sample_train_entities(s1_frame({"US": 10}), 5, "train", False)

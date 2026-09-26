"""s5_score.score_split: streams to disk one shard at a time; --val-only keeps just the held-out entities;
shard size never changes the output; the file appears atomically."""
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import pytest

import config
import pipeline_io as pio
import s5_score


class _Model:
    def predict(self, X):
        return X[:, 0].astype(np.float64)          # prob = first feature, so every row is checkable


@pytest.fixture
def feats(tmp_path):
    n_ent, per = 40, 7
    ids = [f"e{i:03d}" for i in range(n_ent) for _ in range(per)]
    cols = pio.feature_columns(len(pio.feature_spec()[0]))
    rng = np.random.default_rng(1)
    data = {"source1_entity_id": ids, "candidate_entity_id": [f"c{i}" for i in range(len(ids))]}
    for c in cols:
        data[c] = rng.random(len(ids)).astype(np.float32)
    data["label"] = np.zeros(len(ids), dtype=np.uint8)
    pl.DataFrame(data).write_parquet(config.features_path("train", tmp_path), row_group_size=25)
    return tmp_path, cols, pl.DataFrame(data)


def _score(tmp_path, cols, **kw):
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    n = s5_score.score_split("train", tmp_path, out, _Model(), None, False, cols,
                             kw.pop("shard_rows", 1000), kw.pop("val_only", False), False)
    return n, pl.read_parquet(config.scored_path("train", out)), out


def test_scores_every_row_and_matches_model(feats):
    tmp_path, cols, df = feats
    n, scored, out = _score(tmp_path, cols)
    assert n == df.height == scored.height
    got = scored.join(df.select("source1_entity_id", "candidate_entity_id", cols[0]), on=["source1_entity_id", "candidate_entity_id"])
    assert np.allclose(got["prob"].to_numpy(), got[cols[0]].to_numpy())
    assert scored.schema == pl.Schema(pio.SCORED_SCHEMA)
    assert not list(out.glob("*.tmp"))                                   # renamed into place, no leftovers


def test_output_does_not_depend_on_shard_size(feats):
    tmp_path, cols, _ = feats
    _, a, _ = _score(tmp_path, cols, shard_rows=13)
    _, b, _ = _score(tmp_path, cols, shard_rows=100_000)
    assert a.sort("candidate_entity_id").equals(b.sort("candidate_entity_id"))


def test_val_only_scores_just_held_out_entities(feats, monkeypatch):
    tmp_path, cols, df = feats
    held = pl.Series("entity_id", ["e003", "e017", "e039"])
    monkeypatch.setattr(pio, "val_ids", lambda smoke: held)
    n, scored, _ = _score(tmp_path, cols, val_only=True, shard_rows=20)
    assert set(scored["source1_entity_id"]) == set(held)
    assert n == scored.height == 3 * 7

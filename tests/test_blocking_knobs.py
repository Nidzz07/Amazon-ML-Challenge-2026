"""Corpus-relative df ceilings (blocking.common.resolve_df) and the embed_ann pairs reader."""
import polars as pl
import pytest

import config
import pipeline_io as pio
from blocking import embed_ann, rare_token
from blocking.common import CHANNEL_SCHEMA, pop_resolved, resolve_df


def norm(rows: list[dict]) -> pl.DataFrame:
    base = {c: "" for c, t in pio.NORM_SCHEMA.items() if t == pl.String}
    full = [{**base, "name_tokens": [], "addr_tokens": [], "has_addr": True, "script": 1, **r} for r in rows]
    return pl.DataFrame(full, schema=pio.NORM_SCHEMA)


def test_resolve_df_fraction_count_none():
    pop_resolved()
    assert resolve_df("X", 0.2, 93_257) == 18_651  # floored
    assert resolve_df("Y", 20_000, 93_257) == 20_000  # an int is already a count
    assert resolve_df("Z", None, 93_257) is None
    assert pop_resolved()["X"] == {"value": 0.2, "n_docs": 93_257, "resolved": 18_651}
    with pytest.raises(ValueError):
        resolve_df("X", 1.5, 10)


def test_rare_token_ceiling_is_relative_to_the_shard(monkeypatch):
    # Shard = 4 records. "common" (df 4) is kept at 1.0 (-> 4) and dropped at 0.5 (-> 2);
    # "zeta" (df 2) survives both.
    s1 = norm([{"entity_id": "S1-1", "name_tokens": ["common", "zeta"]}])
    pool = norm([{"entity_id": f"S2-{i}", "name_tokens": ["common"] + (["zeta"] if i == 1 else [])} for i in (1, 2, 3)])
    monkeypatch.setattr(config, "RARE_TOKEN_DF_MAX", 1.0)
    assert set(rare_token.run(s1, pool)["candidate_entity_id"]) == {"S2-1", "S2-2", "S2-3"}
    monkeypatch.setattr(config, "RARE_TOKEN_DF_MAX", 0.5)
    assert set(rare_token.run(s1, pool)["candidate_entity_id"]) == {"S2-1"}


def write_pairs(path, rows):
    pl.DataFrame(rows, schema={**CHANNEL_SCHEMA, "split": pl.String}, orient="row").write_parquet(path)


def test_embed_ann_reads_explicit_paths_and_filters_to_shard(tmp_path, monkeypatch):
    smoke_dir, full = tmp_path / "smoke", tmp_path / "full_pairs.parquet"
    smoke_dir.mkdir()
    monkeypatch.setattr(config, "SMOKE_ARTIFACTS_DIR", smoke_dir)
    monkeypatch.setattr(config, "EMBED_ANN_PATH", full)
    s1 = norm([{"entity_id": "S1-1"}])
    pool = norm([{"entity_id": "S2-1"}])

    # Missing file on either side: zero candidates, no error.
    assert embed_ann.run(s1, pool, smoke=True).is_empty()
    assert embed_ann.run(s1, pool, smoke=False).is_empty()

    # A full-scale file must never be read by a smoke run, whatever the shard size.
    write_pairs(full, [("S1-1", "S2-1", 1, 0.9, "train")])
    assert embed_ann.run(s1, pool, smoke=True).is_empty()
    assert embed_ann.run(s1, pool, smoke=False).height == 1

    # Candidates outside this shard's pool, other entities and ranks past EMBED_TOP_K are dropped.
    write_pairs(config.embed_ann_path(True), [
        ("S1-1", "S2-1", 1, 0.9, "train"), ("S1-1", "S2-9", 2, 0.8, "train"),
        ("S1-2", "S2-1", 1, 0.7, "train"), ("S1-1", "S3-1", config.EMBED_TOP_K + 1, 0.1, "train"),
    ])
    out = embed_ann.run(s1, norm([{"entity_id": "S2-1"}, {"entity_id": "S3-1"}]), smoke=True)
    pio.check_schema(out, CHANNEL_SCHEMA, "embed_ann output")
    assert out.rows() == [("S1-1", "S2-1", 1, pytest.approx(0.9))]


def test_embed_ann_empty_shard():
    assert embed_ann.run(norm([]), norm([{"entity_id": "S2-1"}])).is_empty()


def test_tfidf_empty_vector_fallback():
    # Ceiling 0.5 of 4 pool records -> 2: "acme" n-grams (df 3) are dropped, so "acme" gets
    # an empty vector without the fallback. The other records' vectors must not change.
    from blocking.tfidf_index import SparseTopNIndex

    pool = pl.Series("t", ["acme", "acme", "acme zeta", "zeta"])
    queries = pl.Series("t", ["acme", "zeta"])
    off = SparseTopNIndex(0.5, 100, 1, fallback_n=0).fit(pool)
    on = SparseTopNIndex(0.5, 100, 1, fallback_n=5).fit(pool)
    q_off, q_on = off.query_matrix(queries), on.query_matrix(queries)
    assert q_off[0].nnz == 0 and 0 < q_on[0].nnz <= 5
    assert q_on[0][:, : on.vocab.height].nnz == 0  # only fallback columns
    p_off = off._pool_slice_t(0, 4).T.tocsr()
    p_on = on._pool_slice_t(0, 4).T.tocsr()
    for r in (2, 3):  # records with n-grams under the ceiling: bit-identical
        assert (p_off[r] != p_on[r][:, : off.vocab.height]).nnz == 0 and p_on[r].nnz == p_off[r].nnz
    assert (q_off[1] != q_on[1][:, : off.vocab.height]).nnz == 0
    hits = {(int(q), int(c)) for qi, ci, _ in on.search(queries, 10) for q, c in zip(qi, ci)}
    assert {(0, 0), (0, 1)} <= hits  # "acme" query now finds the "acme" pool records
    assert not any(q == 0 for qi, _, _ in off.search(queries, 10) for q in qi)

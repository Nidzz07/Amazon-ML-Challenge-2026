"""s3 reads the embedding file lazily and per shard. The rows a shard sees must equal what the old
read-everything-then-filter code produced, for files with and without split/country columns."""
import polars as pl
import pytest

import s3_featurise as s3


def _pairs():
    rows = []
    for split, country, s1s in (("train", "India", ["a1", "a2"]), ("train", "US", ["b1"]), ("test", "US", ["c1"])):
        for s in s1s:
            for r in (1, 2, 3):
                rows.append((s, f"{s}-cand{r}", r, 1.0 - 0.01 * r, split, country))
    return pl.DataFrame(rows, orient="row", schema=["source1_entity_id", "candidate_entity_id", "channel_rank",
                                                    "channel_score", "split", "country"],
                        ).with_columns(pl.col("channel_rank").cast(pl.UInt16), pl.col("channel_score").cast(pl.Float32))


def _old(df: pl.DataFrame, split: str, s1_ids: pl.Series) -> pl.DataFrame:
    """The pre-change behaviour: read the whole file, filter to the split, then to the shard's Source-1 ids."""
    if "split" in df.columns:
        df = df.filter(pl.col("split") == split)
    df = df.select("source1_entity_id", "candidate_entity_id", pl.col("channel_score").alias("embed_cosine"),
                   pl.col("channel_rank").cast(pl.Float32).alias("embed_rank"))
    return df.filter(pl.col("source1_entity_id").is_in(s1_ids.implode()))


def _key(df):
    return df.sort("source1_entity_id", "candidate_entity_id").rows()


@pytest.mark.parametrize("drop", [[], ["country"], ["split", "country"]])
def test_shard_rows_match_old_behaviour(tmp_path, drop):
    df = _pairs().drop(drop)
    df.write_parquet(tmp_path / "embed_ann_pairs.parquet")
    for split, country, ids in (("train", "India", ["a1", "a2"]), ("train", "US", ["b1"]), ("test", "US", ["c1"])):
        s1_ids = pl.Series("entity_id", ids)
        got = s3._embed_for_shard(s3._load_embed_scores(split, tmp_path), country, s1_ids)
        assert _key(got) == _key(_old(df, split, s1_ids)), (split, country, drop)
        assert got.columns == ["source1_entity_id", "candidate_entity_id", "embed_cosine", "embed_rank"]
        assert got.height == 3 * len(ids)


def test_missing_file_is_none(tmp_path):
    assert s3._load_embed_scores("train", tmp_path) is None
    assert s3._embed_for_shard(None, "US", pl.Series("entity_id", ["x"])) is None


def test_loading_is_lazy(tmp_path):
    _pairs().write_parquet(tmp_path / "embed_ann_pairs.parquet")
    assert isinstance(s3._load_embed_scores("train", tmp_path), pl.LazyFrame)

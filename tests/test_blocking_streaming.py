"""exact_key/rare_token support an optional `sink`: instead of accumulating every internal chunk and
returning one DataFrame, each chunk is handed to `sink` as soon as it is produced and run() returns None.
This is what lets s2_block stream a channel's output straight to disk (one chunk in memory, not the whole
channel) -- see run_channel_to_disk. The two paths must produce exactly the same rows."""
import polars as pl
import pytest

import config
import s2_block
from blocking import exact_key, rare_token


def _sink_rows(module, s1, pool, **kw):
    got = []
    assert module.run(s1, pool, sink=got.append, **kw) is None
    return pl.concat(got) if got else module.run(s1[:0], pool[:0], **kw)  # empty(): reuse schema


def _same_rows(a: pl.DataFrame, b: pl.DataFrame) -> bool:
    key = ["source1_entity_id", "candidate_entity_id", "channel_rank", "channel_score"]
    return a.select(key).sort(key).equals(b.select(key).sort(key))


@pytest.fixture(scope="module")
def smoke_us_shard():
    s1, pool = s2_block.load_shard("train", config.SMOKE_ARTIFACTS_DIR, "US", None, ("entity_id",))
    return s1, pool


def test_exact_key_sink_matches_non_sink(smoke_us_shard):
    s1, pool = smoke_us_shard
    s1e, poole = s2_block.load_shard("train", config.SMOKE_ARTIFACTS_DIR, "US", None, exact_key.COLUMNS)
    direct = exact_key.run(s1e, poole)
    streamed = _sink_rows(exact_key, s1e, poole)
    assert streamed.height == direct.height > 0
    assert _same_rows(direct, streamed)


def test_rare_token_sink_matches_non_sink(smoke_us_shard):
    s1r, poolr = s2_block.load_shard("train", config.SMOKE_ARTIFACTS_DIR, "US", None, rare_token.COLUMNS)
    direct = rare_token.run(s1r, poolr)
    streamed = _sink_rows(rare_token, s1r, poolr)
    assert streamed.height == direct.height > 0
    assert _same_rows(direct, streamed)


def test_channel_supports_streaming_flags_the_right_modules():
    assert s2_block.channel_supports_streaming(exact_key)
    assert s2_block.channel_supports_streaming(rare_token)
    from blocking import addr_tfidf, embed_ann, name_tfidf
    assert not s2_block.channel_supports_streaming(name_tfidf)
    assert not s2_block.channel_supports_streaming(addr_tfidf)
    assert not s2_block.channel_supports_streaming(embed_ann)


def test_run_channel_to_disk_writes_same_rows_run_channel_would(tmp_path, smoke_us_shard):
    """End to end through the real checkpoint writer: run_channel_to_disk's output file must equal
    run_channel + write_atomic's, once the added `bit` column is accounted for."""
    s1, pool = s2_block.load_shard("train", config.SMOKE_ARTIFACTS_DIR, "US", None, exact_key.COLUMNS)
    via_disk = tmp_path / "US_exact_key.parquet"
    row = s2_block.run_channel_to_disk("train", "US", 2, "exact_key", exact_key, s1, pool, True, via_disk)
    direct, _ = s2_block.run_channel("train", "US", 2, "exact_key", exact_key, s1, pool, True, False)
    got = pl.read_parquet(via_disk)
    assert _same_rows(got, direct)
    assert (got["bit"] == direct["bit"]).all() and direct["bit"].unique().to_list() == [1 << 2]
    assert row["pairs"] == got.height
    assert row["entities"] == got["source1_entity_id"].n_unique()

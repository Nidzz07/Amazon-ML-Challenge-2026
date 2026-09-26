"""embed_rank is monotone -1 (lower = better): a pair NOT in the ANN search must read as worse than any real rank.
Filling 0 made it read as better than rank 1 (the pre-v3 bug)."""
import numpy as np
import polars as pl

import features
import s3_featurise


def _frames():
    block = pl.DataFrame({"source1_entity_id": ["a", "a", "b"], "candidate_entity_id": ["x", "y", "x"]})
    s1 = pl.DataFrame({"entity_id": ["a", "b"]})
    pool = pl.DataFrame({"entity_id": ["x", "y"]})
    return block, s1, pool


def test_join_fills_missing_rank_with_sentinel():
    block, s1, pool = _frames()
    embed = pl.DataFrame({"source1_entity_id": ["a"], "candidate_entity_id": ["x"],
                          "embed_cosine": [0.9], "embed_rank": [1.0]})
    out = s3_featurise._join_block(block, s1, pool, embed, None)
    got = dict(zip(zip(out["source1_entity_id"], out["candidate_entity_id"]), out["embed_rank"]))
    assert got[("a", "x")] == 1.0
    assert got[("a", "y")] == features.EMBED_RANK_MISSING
    assert got[("b", "x")] == features.EMBED_RANK_MISSING
    assert out["embed_cosine"].to_list().count(0.0) == 2  # cosine keeps 0: it is monotone +1


def test_no_embed_file_at_all_is_all_sentinel():
    block, s1, pool = _frames()
    out = s3_featurise._join_block(block, s1, pool, None, None)
    assert (out["embed_rank"] == features.EMBED_RANK_MISSING).all()


def test_sentinel_is_worse_than_any_real_rank_and_version_bumped():
    assert features.EMBED_RANK_MISSING > 20  # EMBED_TOP_K = 20
    idx = features.FEATURE_NAMES.index("embed_rank")
    assert features.FEATURE_MONO[idx] == -1
    assert features.FEATURE_VERSION >= 3


def test_np_col_fill_and_absent_column():
    df = pl.DataFrame({"embed_rank": [1.0, None]})
    assert features._np_col(df, "embed_rank", fill=9999.0).tolist() == [1.0, 9999.0]
    assert np.all(features._np_col(pl.DataFrame({"z": [1, 2]}), "embed_rank", fill=9999.0) == 9999.0)

"""s2 subsamples train once; s3 must featurise exactly that entity set and never re-sample.

The chain is sample_train_entities -> candidates_train (s2) -> features_train (s3). If
any stage narrowed or re-drew entities on its own, these three id sets would differ.
"""
import shutil

import polars as pl
import pytest

import config
import pipeline_io as pio
import s2_block
import s3_featurise

N = 3_000
SMOKE_READY = all(config.norm_path(sp, src, config.SMOKE_ARTIFACTS_DIR).exists()
                  for sp, srcs in config.SPLITS.items() for src in srcs if src != "ground_truth")


@pytest.mark.skipif(not SMOKE_READY, reason="smoke norm parquet missing - run s0/s1 --smoke")
def test_sampler_candidates_features_share_one_entity_set(tmp_path):
    src = config.SMOKE_ARTIFACTS_DIR
    s2_block.main(["--smoke", "--input", str(src), "--output", str(tmp_path), "--train-entities", str(N)])
    # s3 reads norm + ground truth from the same dir as the candidates.
    for split, srcs in config.SPLITS.items():
        for s in srcs:
            if s != "ground_truth":
                shutil.copy(config.norm_path(split, s, src), config.norm_path(split, s, tmp_path))
    shutil.copy(config.records_path("train", "ground_truth", src), config.records_path("train", "ground_truth", tmp_path))
    s3_featurise.main(["--smoke", "--input", str(tmp_path), "--output", str(tmp_path), "--splits", "train"])

    s1 = pl.read_parquet(config.norm_path("train", config.SOURCE1_SRC, src), columns=["entity_id", "country"])
    sampled = set(pio.sample_train_entities(s1, N, "train", True))
    cands = set(pl.read_parquet(config.candidates_path("train", tmp_path))["source1_entity_id"].unique())
    feats = set(pl.read_parquet(config.features_path("train", tmp_path), columns=["source1_entity_id"])
                ["source1_entity_id"].unique())
    assert len(sampled) == N
    assert cands == sampled, f"candidates_train: {len(cands - sampled)} extra, {len(sampled - cands)} missing"
    assert feats == sampled, f"features_train: {len(feats - sampled)} extra, {len(sampled - feats)} missing"

    # Test is never sampled: every test Source-1 entity is blocked.
    test_s1 = set(pl.read_parquet(config.norm_path("test", config.SOURCE1_SRC, src), columns=["entity_id"])["entity_id"])
    assert set(pl.read_parquet(config.candidates_path("test", tmp_path))["source1_entity_id"].unique()) == test_s1

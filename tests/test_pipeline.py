"""End-to-end smoke run: s0 -> s7 back to back on --smoke, within BUDGET_SEC (two minutes plus margin).

Every stage reads and writes a tmp dir (--input/--output) so the test never
clobbers artifacts/smoke/. s0 still reads the real smoke TSVs.
"""
import importlib.util
import shutil
import time

import polars as pl
import pytest

import config
import pipeline_io as pio
import s0_ingest
import s1_normalise
import s2_block
import s3_featurise
import s4_train
import s5_score
import s6_assemble
import s7_evaluate

SMOKE_READY = all(config.smoke_raw_path(sp, src).exists() for sp, srcs in config.SPLITS.items() for src in srcs)
STAGES = [s1_normalise, s2_block, s3_featurise, s4_train, s5_score, s6_assemble, s7_evaluate]
BUDGET_SEC = 180


def load_validator():
    spec = importlib.util.spec_from_file_location("validate_submission", config.VALIDATOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not SMOKE_READY, reason="smoke TSVs missing - run make_smoke_sample.py")
def test_smoke_pipeline_end_to_end(tmp_path):
    work = tmp_path / "artifacts"
    io_args = ["--smoke", "--input", str(work), "--output", str(work)]

    timings = {}
    t0 = time.perf_counter()
    t = time.perf_counter()
    s0_ingest.main(["--smoke", "--output", str(work)])
    timings["s0_ingest"] = time.perf_counter() - t
    for stage in STAGES:
        t = time.perf_counter()
        stage.main(io_args + (["--tag", "pytest"] if stage is s7_evaluate else []))
        timings[stage.__name__] = time.perf_counter() - t
    total = time.perf_counter() - t0
    print("\n" + "\n".join(f"  {k:<14} {v:6.1f}s" for k, v in timings.items()) + f"\n  {'total':<14} {total:6.1f}s")
    assert total < BUDGET_SEC, f"smoke chain took {total:.1f}s, budget {BUDGET_SEC}s"

    for split, srcs in config.SPLITS.items():
        for src in srcs:
            if src != "ground_truth":
                pio.check_schema(pl.read_parquet(config.norm_path(split, src, work)), pio.NORM_SCHEMA, f"norm_{split}_{src}")
        pio.check_schema(pl.read_parquet(config.candidates_path(split, work)), pio.CANDIDATES_SCHEMA, f"candidates_{split}")
        pio.check_schema(pl.read_parquet(config.scored_path(split, work)), pio.SCORED_SCHEMA, f"scored_{split}")
        feats = pl.read_parquet(config.features_path(split, work))
        names, _ = pio.feature_spec()
        want = ["source1_entity_id", "candidate_entity_id", *pio.feature_columns(len(names))]
        want += ["label"] if split == "train" else []
        assert feats.columns == want
        assert all(feats[c].dtype == pl.Float32 for c in pio.feature_columns(len(names)))
        if split == "train":
            assert feats["label"].dtype == pl.UInt8

    assert config.model_path(work).exists() and config.calibrator_path(work).exists()
    assert config.report_path("pytest", work).exists()

    # The organisers' validator must PASS on the smoke test submission.
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()
    for src in config.SPLITS["test"]:
        shutil.copy(config.smoke_raw_path("test", src), test_dir / f"test_{src}.tsv")
    errors, _ = load_validator().validate(
        str(config.matching_results_path("test", work)),
        str(config.candidate_pairs_path("test", work)),
        str(test_dir),
        check_ids=True,
    )
    assert errors == []

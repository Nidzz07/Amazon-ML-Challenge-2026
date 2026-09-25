import polars as pl
import pytest

import config
import s0_ingest

HEADER_SRC = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
FIXTURE = {
    ("train", "source1"): HEADER_SRC
    + "S1-1\tRaj Investments LLP\t12 MG Road, Chennai, TN\tIndia\n"
    + 'S1-2\t"""ehpad Club SAS"\t4 Rue Daurat, Saint-Nazaire\tFrance\n'
    + "S1-3\tOrelee's Barbershop\t\tUS\n",
    ("train", "source2"): HEADER_SRC
    + "S2-10\tராஜ் இன்வெஸ்ட்மென்ட்ஸ் எல்எல்பி\t12 எம்ஜி சாலை, சென்னை\tIndia\n"
    + "S2-11\tDréxkor\t85 Wanye Avenue, null\tUS\n",
    ("train", "source3"): HEADER_SRC
    + "S3-20\tएसएस फूड प्राइवेट लिमिटेड\t\tIndia\n",
    ("train", "ground_truth"): "source1_entity_id\tmatched_entity_ids\n"
    + "S1-1\tS2-10,S3-20\n"
    + "S1-2\t\n"
    + "S1-3\tS2-11\n",
    ("test", "source1"): HEADER_SRC + "S1-9\tZephay Labs Inc\t2621 Cotten Road, Tyler, TX\tUS\n",
    ("test", "source2"): HEADER_SRC + "S2-90\tMarina Ecole France Sarl\t63 R. DE DIEPPE, LILLE\tFrance\n",
    ("test", "source3"): HEADER_SRC + "S3-90\tमॉडर्न फाइनेंस\tUdyog Vihar Phase V, Gurugram, HR\tIndia\n",
}


def write_fixture(root):
    for (split, src), text in FIXTURE.items():
        p = config.raw_path(split, src, root)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="\n")


def check_outputs(artifacts_dir, dataset_dir):
    for split, srcs in config.SPLITS.items():
        for src in srcs:
            df = pl.read_parquet(config.records_path(split, src, artifacts_dir))
            want = config.GROUND_TRUTH_COLUMNS if src == "ground_truth" else config.SOURCE_COLUMNS
            assert tuple(df.columns) == want
            assert all(dt == pl.String for dt in df.dtypes)
            assert df.null_count().sum_horizontal().item() == 0
            raw_rows = sum(1 for _ in open(config.raw_path(split, src, dataset_dir), encoding="utf-8")) - 1
            assert df.height == raw_rows


def test_ingest_fixture(tmp_path):
    dataset_dir, artifacts_dir = tmp_path / "dataset", tmp_path / "artifacts"
    write_fixture(dataset_dir)
    results = s0_ingest.run(dataset_dir, artifacts_dir, check_counts=False)
    assert len(results) == 7
    check_outputs(artifacts_dir, dataset_dir)

    s1 = pl.read_parquet(config.records_path("train", "source1", artifacts_dir))
    assert s1.filter(pl.col("entity_id") == "S1-2")["business_name"].item() == '"ehpad Club SAS'
    assert s1.filter(pl.col("entity_id") == "S1-3")["business_address"].item() == ""

    s2 = pl.read_parquet(config.records_path("train", "source2", artifacts_dir))
    assert s2["business_name"][0] == "ராஜ் இன்வெஸ்ட்மென்ட்ஸ் எல்எல்பி"
    assert s2["business_address"][1] == "85 Wanye Avenue, null"  # literal null kept as text

    gt = pl.read_parquet(config.records_path("train", "ground_truth", artifacts_dir))
    assert gt.filter(pl.col("source1_entity_id") == "S1-2")["matched_entity_ids"].item() == ""


def test_ingest_rejects_wrong_row_count(tmp_path):
    write_fixture(tmp_path / "dataset")
    raw = config.raw_path("train", "source1", tmp_path / "dataset")
    with pytest.raises(AssertionError, match="rows != expected"):
        s0_ingest.ingest_file("train", "source1", raw, tmp_path / "out.parquet", expected_rows=999)


def test_ingest_rejects_bad_header(tmp_path):
    raw = tmp_path / "bad.tsv"
    raw.write_text("id\tname\n1\tx\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="columns"):
        s0_ingest.ingest_file("train", "source1", raw, tmp_path / "out.parquet")


@pytest.mark.skipif(not config.SMOKE_DIR.exists(), reason="data/smoke/ not created yet")
def test_ingest_smoke(tmp_path):
    s0_ingest.run(config.SMOKE_DIR, tmp_path, check_counts=False)
    check_outputs(tmp_path, config.SMOKE_DIR)

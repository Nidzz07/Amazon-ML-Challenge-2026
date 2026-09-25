"""All paths and global knobs. No other module may contain a literal path."""
from pathlib import Path

SEED = 42

# src/ -> business_entity_resolution/ -> code/ -> repo root
ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = ROOT / "data"
DATASET_DIR = DATA_DIR / "dataset"
SMOKE_DIR = DATA_DIR / "smoke"  # mirrors DATASET_DIR layout: smoke/{train,test}/{split}_{src}.tsv
ARTIFACTS_DIR = ROOT / "artifacts"
SMOKE_ARTIFACTS_DIR = ARTIFACTS_DIR / "smoke"  # smoke runs never overwrite full-size artifacts
OUTPUT_DIR = ROOT / "output"
VALIDATOR = DATA_DIR / "utils" / "validate_submission.py"

TSV_SEP = "\t"
ENCODING = "utf-8"

SPLITS = {
    "train": ("source1", "source2", "source3", "ground_truth"),
    "test": ("source1", "source2", "source3"),
}

SOURCE_COLUMNS = ("entity_id", "business_name", "business_address", "country")
GROUND_TRUTH_COLUMNS = ("source1_entity_id", "matched_entity_ids")


def raw_path(split: str, src: str, dataset_dir: Path = DATASET_DIR) -> Path:
    return Path(dataset_dir) / split / f"{split}_{src}.tsv"


def records_path(split: str, src: str, artifacts_dir: Path = ARTIFACTS_DIR) -> Path:
    return Path(artifacts_dir) / f"records_{split}_{src}.parquet"


TRAIN_SOURCE1 = raw_path("train", "source1")
TRAIN_SOURCE2 = raw_path("train", "source2")
TRAIN_SOURCE3 = raw_path("train", "source3")
TRAIN_GROUND_TRUTH = raw_path("train", "ground_truth")
TEST_SOURCE1 = raw_path("test", "source1")
TEST_SOURCE2 = raw_path("test", "source2")
TEST_SOURCE3 = raw_path("test", "source3")

RAW_FILES = {(split, src): raw_path(split, src) for split, srcs in SPLITS.items() for src in srcs}

# Measured row counts of the full dataset; s0_ingest asserts against these.
EXPECTED_ROWS = {
    ("train", "source1"): 2_206_821,
    ("train", "source2"): 5_034_616,
    ("train", "source3"): 5_285_603,
    ("train", "ground_truth"): 2_206_821,
    ("test", "source1"): 1_732_544,
    ("test", "source2"): 4_887_273,
    ("test", "source3"): 5_082_316,
}

"""All paths and global knobs. No other module may contain a literal path."""
from pathlib import Path

SEED = 42

# src/ -> business_entity_resolution/ -> code/ -> repo root
ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = ROOT / "data"
DATASET_DIR = DATA_DIR / "dataset"
ARTIFACTS_DIR = ROOT / "artifacts"
SMOKE_ARTIFACTS_DIR = ARTIFACTS_DIR / "smoke"  # smoke runs never overwrite full-size artifacts
# Smoke raw TSVs live flat beside the smoke parquet: artifacts/smoke/smoke_{split}_{src}.tsv
SMOKE_DIR = SMOKE_ARTIFACTS_DIR
OUTPUT_DIR = ROOT / "output"
SMOKE_OUTPUT_DIR = SMOKE_ARTIFACTS_DIR / "output"  # smoke submissions never touch output/
REPORTS_DIRNAME = "reports"
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


def smoke_raw_path(split: str, src: str, smoke_dir: Path = SMOKE_DIR) -> Path:
    return Path(smoke_dir) / f"smoke_{split}_{src}.tsv"


def records_path(split: str, src: str, artifacts_dir: Path = ARTIFACTS_DIR) -> Path:
    return Path(artifacts_dir) / f"records_{split}_{src}.parquet"


def norm_path(split: str, src: str, artifacts_dir: Path = ARTIFACTS_DIR) -> Path:
    return Path(artifacts_dir) / f"norm_{split}_{src}.parquet"


def candidates_path(split: str, artifacts_dir: Path = ARTIFACTS_DIR) -> Path:
    return Path(artifacts_dir) / f"candidates_{split}.parquet"


def features_path(split: str, artifacts_dir: Path = ARTIFACTS_DIR) -> Path:
    return Path(artifacts_dir) / f"features_{split}.parquet"


def scored_path(split: str, artifacts_dir: Path = ARTIFACTS_DIR) -> Path:
    return Path(artifacts_dir) / f"scored_{split}.parquet"


def model_path(artifacts_dir: Path = ARTIFACTS_DIR) -> Path:
    return Path(artifacts_dir) / "model.txt"


def calibrator_path(artifacts_dir: Path = ARTIFACTS_DIR) -> Path:
    return Path(artifacts_dir) / "calibrator.pkl"


def matching_results_path(split: str, out_dir: Path) -> Path:
    # The test files carry the exact names the organisers' validator expects;
    # other splits get a suffix so they can never be mistaken for a submission.
    return Path(out_dir) / ("matching_results.tsv" if split == "test" else f"matching_results_{split}.tsv")


def candidate_pairs_path(split: str, out_dir: Path) -> Path:
    return Path(out_dir) / ("candidate_pairs.tsv" if split == "test" else f"candidate_pairs_{split}.tsv")


def report_path(tag: str, artifacts_dir: Path = ARTIFACTS_DIR) -> Path:
    return Path(artifacts_dir) / REPORTS_DIRNAME / f"report_{tag}.json"


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

SOURCE1_SRC = "source1"
CANDIDATE_SRCS = ("source2", "source3")

# Validation split: held-out Source-1 train entities. S2/S3 stay complete (never
# filtered) so validation sees the same distractor density as test.
VAL_SIZE = 400_000
VAL_ENTITY_IDS = ARTIFACTS_DIR / "val_entity_ids.parquet"

# Smoke sample.
SMOKE_TRAIN_ENTITIES = 50_000
SMOKE_DISTRACTOR_RATIO = 10  # distractor S2/S3 records per true-match record
SMOKE_TEST_ENTITIES = 5_000
SMOKE_TEST_MIN_PER_COUNTRY = 500  # every test country (France included) gets at least this many S1 rows

# Blocking.
# Channel bit positions in candidates.channels (uint8 bitmask).
CHANNELS = ("name_tfidf", "addr_tfidf", "exact_key", "rare_token", "embed_ann")
MAX_CANDIDATES_PER_ENTITY = 60
EXACT_KEY_MAX_BUCKET = 500

# Placeholder decision threshold used only by the S6 stub until the
# expected-F0.5 prefix search replaces it.
STUB_MATCH_THRESHOLD = 0.5

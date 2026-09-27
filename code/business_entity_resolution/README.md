# Business entity resolution: Amazon ML Challenge 2026

Given a Source-1 business record, find every Source-2 / Source-3 record that is the same business.
Records are noisy: different scripts (Latin / Devanagari / Tamil / ...), reordered addresses, legal-suffix
drift, house-number mutation. The score is macro F0.5 per Source-1 entity.

The pipeline is a chain of stages that pass Parquet files through `artifacts/`. **The model never sees raw
text**: stage S3 turns each candidate pair into a fixed-width numeric vector, and S4/S5 are plain tabular ML.

```
S0 ingest -> S1 normalise -> S2 block ----------> S3 featurise -> S4 train -> S5 score -> S6 assemble -> S7 evaluate
                      \-> S2a embed (GPU) -> embed_ann_pairs.parquet (5th blocking channel + 2 features) /
```

## Environment

| | |
|---|---|
| Python | Developed and tested on **3.13.9** (Windows 11). The Kaggle embedding run used **3.12**. The roadmap targeted 3.11; **3.11 was not tested.** |
| Dev hardware | 16 cores, 24 GB RAM, RTX 4050 6 GB |
| GPU stage | Kaggle **T4 x2** (`s2a_embed.py`) |

```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-embed.txt                # only for the GPU embedding stage; needs a CUDA torch
export PYTHONUTF8=1                                  # Indic text must never be read as cp1252
```

Windows note: Smart App Control can refuse to load unsigned binaries (it blocked polars once during development,
"An Application Control policy has blocked this file"). If imports hang or fail with that message, run in WSL2 or
on Kaggle.

## Data layout

```
data/dataset/train/{train_source1,train_source2,train_source3,train_ground_truth}.tsv
data/dataset/test/{test_source1,test_source2,test_source3}.tsv
data/utils/validate_submission.py     # the organisers' validator, unmodified (also run by the s6 submission gate)
artifacts/                            # every intermediate Parquet; artifacts/smoke/ for the smoke sample
output/                               # matching_results.tsv, candidate_pairs.tsv
```

All paths are relative to the project root, which `config.py` takes to be three directories above `src/`
(`src/` -> `business_entity_resolution/` -> `code/` -> root).

**Inside the submission zip** the project root is the zip root. Unpack it, then put the organisers' data next
to `code/` and `output/`:

```
<zip root>/
  code/business_entity_resolution/{src/, README.md, requirements.txt, requirements-embed.txt}
  data/dataset/{train,test}/...        # the organisers' TSVs (not shipped in the zip)
  data/utils/validate_submission.py    # the organisers' validator (not shipped in the zip)
  output/                              # the submitted TSVs; a re-run of s6 overwrites them
```

`requirements.txt` and `requirements-embed.txt` sit next to this README; install them from here.

## Running it (full data)

Run from `code/business_entity_resolution/src/`. Every stage takes `--smoke` to read/write `artifacts/smoke/`, and
`--input DIR` / `--output DIR` to override the directories.

```bash
python s0_ingest.py                  # raw TSV -> records_{split}_{src}.parquet (asserts exact row counts)
python s1_normalise.py               # -> norm_{split}_{src}.parquet (romanisation, suffixes, address parsing)
python validation_split.py           # holds out the seeded validation entities -> artifacts/val_entity_ids.parquet

python s2a_embed.py                  # GPU, ~hours: -> artifacts/embed_ann_pairs.parquet (see below). Do this BEFORE s2/s3.
python s2_block.py [--splits train test] [--channels ...] [--resume]   # -> candidates_{split}.parquet, checkpointed in artifacts/s2_parts/
python s3_featurise.py               # -> features_{split}.parquet, streamed by country shard and chunk
python s4_train.py                   # -> model.txt, model.meta, calibrator.pkl
python s5_score.py --splits test                 # -> scored_test.parquet
python s5_score.py --splits train --val-only     # scores only the held-out validation entities (what S7 evaluates)
python s6_assemble.py                # -> output/matching_results.tsv, output/candidate_pairs.tsv (test)
python s7_evaluate.py                # -> artifacts/reports/report_<tag>.json (macro F0.5, by country, by bucket)
```

Smoke run (small, minutes): `python make_smoke_sample.py` once, then each stage with `--smoke`.

### Splitting S2 blocking across machines

S2 checkpoints every (country, channel) output to `artifacts/s2_parts/<split>/<country>_<channel>.parquet`, and
`manifest.json` in that folder fingerprints the shared inputs and each channel's own settings. That lets the
work be split:

1. Every machine runs the **same commit** on the **same** `norm_{split}_source*.parquet` files.
2. Each machine computes its share, either whole channels with
   `python s2_block.py --splits train --channels name_tfidf addr_tfidf --resume`, or a single shard with
   `python compute_channel_shard.py addr_tfidf US --splits train`.
3. Copy the resulting `<country>_<channel>.parquet` files into one machine's `artifacts/s2_parts/<split>/`.
   `compute_channel_shard.py` also prints a `"channels"` JSON entry: merge it into that machine's
   `manifest.json`, or `--resume` treats the copied part as stale and recomputes it.
4. On that machine run `python s2_block.py --splits <split> --resume`. It reuses every part whose settings
   match, computes anything missing, and runs the per-country union + cap once all five channels are present.

Never run S2 on a split **without** `--resume` once parts from another machine are in place: that recomputes
every channel listed (all five by default) and deletes the copied parts.

## The embedding stage (S2a)

`s2a_embed.py` embeds `business_name + name_roman` and `business_address + addr_roman` **separately** with
`intfloat/multilingual-e5-base` (MIT, 768-d, fp16, max 64 tokens), then finds each Source-1 entity's top-20 pool
neighbours with an **exact chunked GPU search** (score = mean of the name and address cosines). It works per
(split, country) shard, uses all visible GPUs, and writes resumable chunk files as it goes.

Output: `artifacts/embed_ann_pairs.parquet`, one file for every split and country:
`source1_entity_id` (str), `candidate_entity_id` (str), `channel_rank` (u16, 1..20), `channel_score` (f32),
`split` (str), `country` (str). The production file has 78,787,300 rows (584 MB), exactly 20 per Source-1 entity.

It feeds the pipeline twice: the `embed_ann` blocking channel (S2) and the `embed_cosine` / `embed_rank` features (S3).
A pair absent from the file gets cosine 0 and rank 9999 (the rank feature is monotone-decreasing, so "missing" must
read as worse than any real rank). **Embeddings are all-or-nothing across train and test**: a file covering only one
split would give the model real values in training and constants at inference.

Kaggle recipe (Kaggle CPU/GPU images already ship torch): clone the branch, `pip install -r requirements.txt`,
symlink the attached dataset to `data/dataset`, run S0 + S1, then S2a with the accelerator set to **GPU T4 x2**. Commit
the notebook so it survives the browser closing. Delete the norm/records Parquet and the `data/dataset` symlink at the
end so the output holds only the embeddings file.

## Model

- **LightGBM**, binary objective, lr 0.05, 63 leaves, up to 500 rounds with early stopping (50).
- **Monotone constraints** on every similarity feature (`features.FEATURE_MONO`): more overlap can never lower the
  match probability. Country is deliberately not a feature (France has no training data).
- **Training data**: every train entity except the held-out validation entities. Per entity: all positives plus the
  top `ceil(2 x n_pos)` negatives by `prior_score` (floor 3). Entities are split 70/20/10 into train / early-stopping /
  calibration by a stable hash of the id, never by pair.
- **Calibration**: isotonic regression on the 10% slice that is *not* resampled (every capped candidate of those
  entities), so it learns the true inference prevalence rather than the 2:1 training mix. `model.meta` records
  Brier and ECE from an entity-level 2-fold cross-fit.
- `FEATURE_VERSION` (currently 3) is written to `model.meta`; S5 refuses to score if it differs from `features.py`.
- Seed 42 throughout (`config.SEED`).

## Resources (measured on the dev box unless marked)

| Stage | Measured |
|---|---|
| S2a embed (Kaggle T4 x2) | ~2,100 texts/s including search; train + test ~45M texts, ~7 h wall |
| S3 featurise | ~9-12k pairs/s on the smoke sample; ~600-1,000 MB per 1M pairs (smoke) |
| S4 train | streaming; ~0.48 GB per million kept rows + ~2.2 GB fixed (measured on 12M and 24M synthetic rows); full-scale peak ~12 GB is an **estimate** |
| S5 score | ~200k rows/s, **2.7 GB flat** at 1M-row shards |
| S3 embed read | 1.8-2.7 GB per shard (was 6.5 GB) |

## Tests

The test suite lives in the project repository (`tests/`) and is not part of the submission zip. In the
repository: `python -m pytest tests -q` (needs `data/utils/validate_submission.py` and the smoke sample).

## Known limitations

- France has no ground truth, so nothing here measures French recall; the transfer test (US <-> India) is only a proxy.
- Blocking recall figures were measured on an India sample and on the smoke sample; the candidate-cap analysis is
  summarised in `Documentation_template.md` (Appendix B).
- Several legacy scaffold scripts remain in `src/` (`build_training_set.py`, `train.py`, `calibrate.py`, `score.py`,
  `transfer_test.py`); the pipeline stages above replace them and nothing imports them.

# Amazon ML Challenge 2026 — Business Entity Resolution

As of: 2026-09-25. Due: 2026-09-27, 23:59 IST.

The scored quantity is macro-averaged F0.5 over 1,732,544 test Source-1 entities, and it collapses to one expression per entity: **F = 1.25c / (0.25k + m)**, where c is correct predictions, k is true matches and m is how many you predicted. Every decision below exists to maximise that expression before the deadline.

Team of four working in parallel: Nidhi (candidate generation + assembly), Tanuj (matching model), Parth (normalisation + transliteration), Krrish (features + evaluation).

---

## Scoreboard math

Substituting P = c/m and R = c/k into Fβ with β = 0.5 cancels almost everything:

```
F_0.5 = (1.25 * P * R) / (0.25*P + R) = (1.25 * c) / (0.25*k + m)
```

c = matches you got right, k = true matches for that entity, m = how many you predicted. Checked against the organisers' own worked example: predicting 3 when 2 of them are right and k = 2 gives 1.25·2/(0.5+3) = 0.714, exactly the number printed in the problem statement. The metric is linear in c and has a constant 0.25k offset in the denominator, which produces three consequences that most teams will miss.

**1. The optimal threshold is not 0.5, and it is not constant.** Ask whether to add one more candidate that is correct with probability q. Expected score after adding is 1.25(c+q)/(0.25k+m+1), before adding it is 1.25c/(0.25k+m). Adding is worth it precisely when

```
q > c / (0.25*k + m)
```

The bar rises as the list grows. At the dataset's mean k = 3.46, with every prediction so far correct:

| Candidates already predicted | Probability the next one must clear |
| --- | --- |
| 1 | 0.536 |
| 2 | 0.698 |
| 3 | 0.776 |
| 4 | 0.822 |
| 5 | 0.852 |

A fixed global threshold of 0.5 over-predicts on every entity past the first match. This alone is worth several points.

**2. Singletons are free points, and they need their own decision.** 5.58% of training entities have no match at all; an empty prediction scores a full 1.0 and any prediction scores 0.0. The marginal rule above assumes k > 0 and breaks at k = 0, so the choice between predicting nothing and predicting one thing is a separate comparison: P(k = 0) against the expected score of the one-item list.

**3. The selection layer should be a search, not a threshold.** Sort an entity's candidates by calibrated probability descending, then evaluate expected F0.5 for every prefix length m = 0 to n and take the best. Plug in ĉ = Σ(i≤m) qᵢ and k̂ = Σ over all candidates qᵢ; that approximation is close enough at 1.7M entities and costs microseconds each. Krrish calibrates the residual bias on the validation split.

This selection layer is Nidhi's, it is independent of whichever model produces q, and it is the single highest-leverage component in the pipeline.

---

## Data profile

Measured directly from the files, not assumed. Naive comparison is 1.73M × 9.97M ≈ 1.7×10¹³ pairs, so blocking is not a stage, it is the problem.

| File | Rows | Size |
| --- | --- | --- |
| train_source1.tsv | 2,206,821 | 210 MB |
| train_source2.tsv | 5,034,616 | 489 MB |
| train_source3.tsv | 5,285,603 | 504 MB |
| train_ground_truth.tsv | 2,206,821 | 127 MB |
| test_source1.tsv | 1,732,544 | 175 MB |
| test_source2.tsv | 4,887,273 | 509 MB |
| test_source3.tsv | 5,082,316 | 506 MB |

**Each Source-2 / Source-3 record belongs to at most one Source-1 entity.** All 7,638,365 matched IDs in the ground truth are distinct — zero reuse across the 2.2M rows. The true structure is many-to-one, which licenses a global uniqueness constraint at assembly and a set of competition features during scoring. This is the strongest structural prior in the dataset and it is not stated anywhere in the problem statement.

**Label shape.** Mean 3.46 matches per entity, mode 3, maximum 11. 5.58% are singletons (123,247 rows). Of the matched IDs, 3,693,619 come from Source 2 and 3,944,746 from Source 3. Roughly 26% of all Source-2/3 records match nothing at all and exist purely as distractors.

**Country.** Training is US (1,323,633 S1) and India (883,188 S1). Test adds France: 259,452 S1 entities, 15% of the test set, with no training data whatsoever. Every sampled positive pair was same-country, but **verify this before relying on it** — it is Hour-0 task 1, and if even 0.1% of true pairs cross countries, country becomes a soft feature instead of a hard partition.

**Missing addresses.** 168,967 train-S2 and 175,916 train-S3 rows have an empty address; 129,408 and 136,098 in test. Those records can only be matched on name.

### Noise catalogue, from real positive pairs

Every pattern below was read off actual matched groups in the training data.

- **Cross-script transliteration.** `Raj Investments LLP` matches its Tamil transliteration. `Ss Food Private Limited` matches its Devanagari transliteration. States appear as native-script names, official codes (`TN`), and full English names (`Tamil Nadu`). Latin-only string similarity scores these near zero. Devanagari, Tamil and Kannada all appear; assume more scripts exist.
- **Name-blind matches.** `Maure Williams Colombier Inc` at 85 Wayne Avenue matches `Dréxkor` at `85 Wanye Avenue` — no name overlap at all, address only. A name-only blocker has a hard recall ceiling.
- **Address-blind matches.** The same entity also matches two records with completely empty addresses. You need both channels, independently.
- **Injected accents.** `Enterprises` → `Énterprises`, `Boral` → `Bóral`. Strip accents for matching but keep the original.
- **Character-level typos.** `Enterpires`, `Etrepndiels`, `Wanye`, `AKON`, `Wilblims`, `Ponr`. Edit distance still works; token equality does not.
- **Abbreviation drift.** `St` / `Street` / `SAINT`, `Rd` / `Road`, `Ave` / `Avenue`, `IL` / `Illinois`, `Pvt` / `Private`, `Ltd` / `Limited`.
- **Component reordering.** `630 45th Terrace, Kansas City, MO` appears as `KANSAS CITY, MO, 630 45ND TERRACE, null`. Token-set similarity survives this; sequence similarity does not.
- **Literal `null` tokens** embedded mid-address, plus `PO BOX` insertions and house-number mutations (`630 45th` → `45ND`, `1056` → `1056c`).
- **Domain names as business names.** `maurewilliamscolombier.com` matches `Maure Williams Colombier Inc`. Strip the TLD and split the stem.
- **Legal-suffix drift.** The suffix is present, absent, or moved in roughly a third of positives. France brings `SARL`, `SAS`, `SASU`, `SCI`, `EURL`, none of which appear in training.

---

## Architecture

Seven stages, each a separate script reading and writing Parquet. Nothing is held in memory across stages, so any stage can be re-run alone on any machine — that is what makes four-way parallelism possible on a 53-hour clock.

`ingest → normalise → block → featurise → train → score → assemble`, with `evaluate` hanging off the side.

| Stage | Script | Owner | Reads | Writes |
| --- | --- | --- | --- | --- |
| S0 Ingest | `s0_ingest.py` | Nidhi | raw `.tsv` | `records_{split}_{src}.parquet` |
| S1 Normalise | `s1_normalise.py` | Parth | S0 output | `norm_{split}_{src}.parquet` |
| S2 Block | `s2_block.py` | Nidhi | S1 output | `candidates_{split}.parquet` |
| S3 Featurise | `s3_featurise.py` | Krrish | S1 + S2 output | `features_{split}.parquet` |
| S4 Train | `s4_train.py` | Tanuj | S3 train output + labels | `model.txt`, `calibrator.pkl` |
| S5 Score | `s5_score.py` | Tanuj | model + S3 test output | `scored_{split}.parquet` |
| S6 Assemble | `s6_assemble.py` | Nidhi | S5 output | `matching_results.tsv`, `candidate_pairs.tsv` |
| S7 Evaluate | `s7_evaluate.py` | Krrish | S6 output + ground truth | `report_{tag}.json` |

### Why the stages sit where they do

**Blocking is a union of five independent channels**, because no single key recovers every positive pair. Each channel emits `(source1_entity_id, candidate_entity_id, channel_id, channel_rank, channel_score)`; the union is deduplicated and capped per entity. A pair retrieved by three channels is far more likely to be a true match than one retrieved by one, so the channel vote count survives into the feature set.

1. Character 3–4-gram TF-IDF cosine on the normalised name, top 20 per entity.
2. The same on the normalised address, top 20 per entity.
3. Exact-key blocks: `(street_number, city)`, `(postcode, street_number)`, `(name_acronym, city)`.
4. Rare-token inverted index — for each entity, look up its lowest-document-frequency name and address tokens.
5. Multilingual embedding ANN, top 20 — the only channel that crosses scripts natively.

**Country partitions the index** if Hour-0 task 1 confirms it. That turns one 1.7×10¹³ problem into three smaller ones and makes France a self-contained shard.

**`candidate_pairs.tsv` is whatever S2 finally hands to S5.** The specification is explicit: it is the last filtering stage, not an early pass. If the per-entity cap is applied inside S2, the capped set is what gets written. Every ID in `matching_results.tsv` must appear in it, and the organisers' validator checks this.

**The model never sees raw text.** S3 turns each pair into a fixed-width numeric vector; S4 and S5 are pure tabular ML. This is what lets Krrish and Tanuj work without waiting for each other — they agree on the feature count and order, and nothing else.

**Assembly enforces the many-to-one constraint** discovered in the data profile: each Source-2/3 record may be claimed by only one Source-1 entity. Resolve conflicts greedily by descending probability, then run the expected-F0.5 prefix search per entity.

---

## Repo and file contracts

The repo mirrors the required zip structure from day one, so packaging at the end is a copy rather than a reorganisation.

```
amazon-ml-2026/
├── code/business_entity_resolution/
│   ├── src/
│   │   ├── config.py          # all paths and knobs, no hard-coded literals elsewhere
│   │   ├── s0_ingest.py ... s7_evaluate.py
│   │   ├── normalise.py       # Parth
│   │   ├── translit.py        # Parth
│   │   ├── lexicon/           # Parth: mined + hand-written mapping tables
│   │   ├── blocking/          # Nidhi: one module per channel
│   │   ├── features.py        # Krrish
│   │   ├── metric.py          # Krrish
│   │   ├── model.py           # Tanuj
│   │   └── assemble.py        # Nidhi
│   ├── README.md
│   └── requirements.txt
├── output/                    # the two scored TSVs, gitignored
├── artifacts/                 # parquet intermediates, gitignored
├── data/                      # raw dataset, gitignored
├── tests/
├── Documentation_template.md
└── utils/validate_submission.py   # copied from student_resource, unmodified
```

### The four schemas that must not drift

These are the entire interface between the four of you. Anyone may change code inside their stage freely; changing a column name here requires telling the other three.

**`records_{split}_{src}.parquet`** — `entity_id` str, `business_name` str, `business_address` str, `country` str.

**`norm_{split}_{src}.parquet`** — `entity_id` str, `name_norm` str, `name_roman` str, `name_tokens` list[str], `name_acronym` str, `addr_norm` str, `addr_roman` str, `addr_tokens` list[str], `street_num` str, `city_norm` str, `state_canon` str, `postcode` str, `country` str, `has_addr` bool, `script` uint8.

**`candidates_{split}.parquet`** — `source1_entity_id` str, `candidate_entity_id` str, `channels` uint8 bitmask, `n_channels` uint8, `best_rank` uint16, `prior_score` float32.

**`features_{split}.parquet`** — the two ID columns, then `f000 ... fNNN` float32 in the exact order of `features.FEATURE_NAMES`, plus `label` uint8 on train splits only.

`features.FEATURE_NAMES` is the single source of truth for feature order and carries a `FEATURE_VERSION` integer. Tanuj's model file records the version it was trained against and `s5_score.py` refuses to run on a mismatch. That one guard prevents the most common way a four-person ML pipeline silently produces garbage.

### Git and artifact discipline

- Branch per track: `track/blocking`, `track/model`, `track/normalise`, `track/features`. Merge to `main` only when the stage runs end to end on the 50k smoke sample.
- `.gitignore` covers `data/`, `artifacts/`, `output/`, `*.parquet`, `*.txt` model files. **Never commit an artifact** — a 2 GB push will cost you an hour you do not have.
- Google Drive holds artifacts only: `/amazon-ml-2026/artifacts/<stage>/<date-hhmm>/`. Name every upload with the git commit SHA that produced it. An artifact whose producing commit is unknown is worthless.
- Everything is seeded and deterministic. `config.py` sets `SEED = 42` and every stage reads it. Two people running the same commit on the same input must get byte-identical output, or debugging becomes impossible.

### The smoke sample

Before anything else, Nidhi cuts a 50,000-entity slice — `smoke_*` files with the full schema — that runs the entire pipeline in under two minutes on the weakest laptop. Every track develops against it. Full-scale runs happen only at gates.

---

## System setup

| Machine | Spec | Role |
| --- | --- | --- |
| Tanuj | Ryzen 7, 24 GB DDR5, RTX 4050 6 GB | Heavy compute box. Full-scale blocking runs and GPU embedding jobs. |
| Parth | i5-13th gen, 16 GB, RTX 4050 6 GB | Normalisation at full scale; second GPU for embedding shards. |
| Krrish | i5-13th gen, 16 GB, RTX 3050 4 GB | Feature development and evaluation harness. |
| Nidhi | Ryzen 5 5500U, 15.3 GB usable, integrated Radeon, 136 GB free | Integrator: design, assembly, submission. No CUDA. |

**One problem worth naming.** Nidhi owns blocking, the most compute-hungry stage, on a low-TDP mobile chip with no usable GPU. Design and debug on the smoke sample locally, run full-scale passes on Tanuj's box or a Kaggle CPU notebook (30 GB RAM, twice her laptop).

### Environment, identical on all four machines

Python **3.11** — not 3.12 or 3.13, several wheels still lag. One virtual environment per person, `requirements.txt` pinned and committed at Gate 0.

```
polars pyarrow pandas numpy scipy scikit-learn
lightgbm rapidfuzz jellyfish python-Levenshtein
indic-transliteration aksharamukha regex unidecode
faiss-cpu sparse-dot-topn
sentence-transformers transformers
tqdm orjson pytest
```

GPU machines add PyTorch separately: `pip install torch --index-url https://download.pytorch.org/whl/cu121`.

### Four Windows problems that will cost hours if unhandled

1. **Encoding.** Windows Python defaults to cp1252, which silently mangles Devanagari, Tamil and Kannada. Set `PYTHONUTF8=1` as a user environment variable on every machine, and pass `encoding="utf-8"` explicitly on every file open regardless.
2. **OneDrive.** Do not put `artifacts/` or `data/` under `Documents` or `Desktop` if OneDrive syncs them. Keep the repo somewhere like `C:\ml2026\`.
3. **Long paths.** Enable them: `Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -Value 1` in an admin PowerShell, then reboot.
4. **Tabs.** Read every file with `sep="\t"` and write with `sep="\t"`. Reading without the explicit separator produces one column and fails silently.

### Disk budget, and the one thing that will not fit

Parquet intermediates run roughly 40–50 GB. What does **not** fit is a materialised feature matrix: 1.73M test entities × 60 candidates × ~60 float32 features is about 25 GB for test alone.

So **S3 and S5 are fused and sharded at inference**: compute features for a shard of roughly 5M pairs, score it immediately, write only `(source1_entity_id, candidate_entity_id, prob)`, discard the features. Only training data is materialised, and only after negative subsampling — keep all 7.6M positives plus hard negatives at 2:1, on a 40% entity subsample, near 15M rows and 4 GB.

### Kaggle, for the GPU stages

Upload the dataset once as a **private Kaggle Dataset**; it mounts read-only at `/kaggle/input/` in every notebook. Free tier gives a T4, sessions up to 9 hours, roughly 30 GPU-hours/week/account. Write to `/kaggle/working/` (20 GB), push results as a new dataset version. **Checkpoint every 500k records** — free sessions die without warning.

### Licence compliance

Final model must be MIT/Apache-2.0, ≤8B parameters. Compliant choices: LaBSE (Apache-2.0, 471M), multilingual-e5-small/base (MIT, 118M/278M), BGE-M3 (MIT, 568M). LightGBM is MIT. Record the chosen model and licence in the methodology document.

**No external lookups of any kind.** No geocoding, no business registries, no entity-resolution APIs, no scraped reference data. Detection means immediate disqualification.

---

## Track A — Nidhi: candidate generation and assembly

You own the recall ceiling and the two files that get scored.

### A0 — Hour 0 to 3, blocking everyone else

1. **Verify the country assumption.** Join ground truth to the source files and count matched pairs whose country differs from their Source-1 entity's. Post the number immediately — three people are waiting on it.
2. **Repo bootstrap.** Folder structure, `config.py`, `.gitignore`, `requirements.txt`, four branches, push.
3. **`s0_ingest.py`.** TSV to Parquet with the fixed schema. Verify round-trip row counts.
4. **Validation split.** 400,000 Source-1 training entities held out, seeded. The Source-2/3 pool stays **complete** for validation — records belonging to training entities act as distractors, exactly the test-time situation.
5. **Smoke sample.** 50,000 entities, full schema, whole pipeline under two minutes.
6. **Stub every stage.** `s1` through `s7` as runnable no-ops with correct schemas.

Gate 0 closes when all four branches exist, the smoke sample runs end to end through the stubs, and the country number is posted.

### A1 — Blocking channels

Five channels under `src/blocking/`, each emitting `(source1_entity_id, candidate_entity_id, channel_id, channel_rank, channel_score)`.

- **`name_tfidf`** — char 3–4-gram TF-IDF over `name_roman`, top 20/entity, per country shard. `sparse_dot_topn`, chunks of 20k. If a shard runs past 40 min, try `TruncatedSVD(256)` + faiss IVFFlat.
- **`addr_tfidf`** — same over `addr_roman`, top 20. Skip rows where `has_addr` is false.
- **`exact_key`** — `(street_num, city_norm)`, `(postcode, street_num)`, `(name_acronym, city_norm)`. Drop buckets over 500 records.
- **`rare_token`** — inverted index, doc frequency 2–5000, union each entity's 3 rarest tokens' postings.
- **`embed_ann`** — Tanuj's embeddings, faiss top 20. Plugs in at Gate 3; pipeline must work without it.

**Union and cap.** Dedup on the pair, set `channels` bitmask and `n_channels`, rank-weighted `prior_score`, keep top 60/entity. This capped set is `candidate_pairs.tsv`.

### A2 — The measurement that matters most

On validation: **recall of true pairs present in the candidate set**, by country and by channel, plus drop-one channel contribution. Target 97%+ recall at median 60 candidates/entity. Measure before Tanuj trains anything.

### A3 — Assembly

1. **Uniqueness constraint.** Group scored pairs by `candidate_entity_id`, keep only the highest-probability claim. Compare a hard version against a soft margin-based version on validation.
2. **Expected-F0.5 prefix search.** Sort candidates by probability descending, compute expected F0.5 for every prefix 0..n using ĉ = Σ(i≤m)qᵢ and k̂ = Σ all qᵢ, take the argmax. m=0 must be a genuine option.

### A4 — Output and validation

`sep="\t"`, `encoding="utf-8"`, no quoting, no index. Before every upload check: exactly 1,732,544 rows + header, no dupes, empty string (never NaN) for singletons, no dupe IDs in a list, every result ID present in that entity's candidate list, every ID exists in test files, no `S1-` IDs in any list. Then run `validate_submission.py` and require `PASS`.

---

## Track B — Tanuj: the matching model

You produce a **calibrated** probability that a pair is a true match. Calibration is not optional — Nidhi's selection layer compares probabilities against metric-derived thresholds.

### B1 — Training data construction

Keep all 7.6M positives. Draw negatives **only from blocking output** (highest-ranked non-matches) — random negatives are trivially separable and inflate validation without helping the leaderboard. Ratio 2:1, 40% entity subsample, ~15M rows, ~4GB. **Split by entity, never by pair.**

### B2 — The model

LightGBM, binary objective, on Krrish's features. Start 500 trees, lr 0.05, 63 leaves.

**Monotonic constraints** on every similarity feature (more overlap can never reduce match probability) — use `monotone_constraints`. **Keep country out of the main model** by default; train a variant with it and compare on the transfer test, since France has no training data.

### B3 — Calibration

Isotonic regression on a held-out slice unused for tree-fitting or threshold tuning. Report reliability curve + Brier score. Confirm probabilities near 0.7 are correct ~70% of the time — that's where the selection layer decides most cases.

### B4 — The France proxy

Train on US-only, evaluate on India-only, and reverse. The in-country vs cross-country drop estimates the France impact. Run at Gate 2, report both numbers.

### B5 — The embedding channel

- Model: **LaBSE** (Apache-2.0, 471M) or **multilingual-e5-base** (MIT, 278M). Both handle Devanagari, Tamil, Kannada, French.
- Embed `name_roman` **and** raw `business_name` (native script carries signal). Separate vectors for name and address.
- Normalise vectors, faiss index per country shard, top 20. India shard ~4.7M × 384 dims ≈ 7GB fp32.
- Deliver ANN pairs (Nidhi's 5th channel) and cosine similarity (Krrish's feature).
- **Checkpoint every 500k records.**

### B6 — Only if Gate 3 lands early

Cross-encoder over top 5 candidates/entity. Highest ceiling, easiest way to run out of time. Do not start before the full pipeline has a validated submission.

---

## Track C — Parth: normalisation and transliteration

~40% of the data is Indian and many positives are written in different scripts. Every string metric scores these at zero without this layer. Everything you write is a **pure function**: string in, string out, no I/O, fully unit-testable.

### C1 — Normalisation pipeline

In order: NFKC normalisation, lowercase, strip accents into a separate field (keep accented original), collapse whitespace/punctuation, remove junk tokens (`null`, `n/a`, `-`, `--`, `<<`, `nil`, empty parens). **Always emit both normalised and original.**

### C2 — Lexicons

Hand-written tables under `src/lexicon/`:

- **Legal suffixes** by country (US: inc/llc/corp/ltd/plc/lp/llp; India: pvt/private/ltd/llp/& co; France: sarl/sas/sasu/sa/sci/eurl/snc). Emit suffix-stripped name + a flag of what was removed.
- **Street abbreviations**: st/street/saint, rd/road, ave/avenue, blvd, ln, dr, ct, sq, hwy, ste/suite, apt, fl/floor, po box.
- **State/region canonicalisation**, three-way: full name, code, native script. All US states + DC. All 28 Indian states + 8 UTs in Devanagari, Tamil, Kannada, Telugu, Bengali, Gujarati, Malayalam, Marathi, Punjabi, Odia. France: handle generically (no training data).
- **Business-word glossary**: food, investments, properties, marketing, finance, infratech, enterprises — in each script.

### C3 — Script detection and romanisation

Detect script by Unicode block, flag it. Romanise with `indic-transliteration` or `aksharamukha` (both MIT, offline). Write to `name_roman`/`addr_roman`. Fold diacritics, collapse doubled vowels after romanising — exact equality won't happen, let fuzzy matching close the gap.

### C4 — The mined lexicon (the differentiator)

The ground truth gives 2.2M known-same-entity groups, often with the same name in Latin and native script.

1. Tokenise the Source-1 name and each matched name per group.
2. Where one side is native-script and the other Latin with close token counts, record token-position co-occurrence.
3. Score by pointwise mutual information, keep support ≥ 20.
4. Result: a data-derived mapping (e.g. native-script tokens → "private", "llp", "food").

Apply this lookup **before** generic romanisation, fall back to the library for unknowns. Same technique mines Latin abbreviation pairs. This uses only provided training data — state that explicitly in the write-up.

### C5 — Address parsing

Extract `street_num`, `city_norm`, `state_canon`, `postcode`, token list. Be forgiving of reordering, house-number mutation, landmark references (`Near SBI ATM`).

### Acceptance test — build this first

30 real positive pairs from the noise catalogue above; assert each scores above a similarity floor after normalisation, before writing any implementation.

---

## Track D — Krrish: features and evaluation

**Write `metric.py` first.** Thirty lines, unblocks everyone.

### D1 — metric.py

Macro F0.5 via `F = 1.25c / (0.25k + m)`. Rules: k=0 → 1.0 for empty prediction, 0.0 for any prediction. Singletons included in the average. Missing entity in prediction file → raise, not silent zero. Average over entities, never pairs.

**Verify against the worked example**: prediction `[S2-00047, S2-00193, S3-00812]`, truth `[S2-00047, S3-00812]` must return exactly 0.714. Commit as a unit test.

### D2 — Feature library

`featurise(s1_rows, cand_rows, context) -> np.ndarray` float32, columns in exact `FEATURE_NAMES` order (name + monotonic direction per feature).

**Name features** (on `name_norm` and `name_roman`): rapidfuzz ratio/partial/token-sort/token-set, Jaro-Winkler, char 3/4-gram Jaccard, token Jaccard/containment, IDF-weighted token cosine, acronym match, first-token match, prefix ratio, length ratio, suffix-stripped ratio + agreement flag, digit-token equality, domain-stem match, script-pair flag.

**Address features**: same family over `addr_norm`/`addr_roman`, plus exact-match flags for `street_num`/`postcode`/`state_canon`/`city_norm`, numeric-token Jaccard, `has_addr` flags both sides.

**Context features**: `n_channels` + per-channel flag, rank within entity per channel + reciprocal rank, margin to entity's best candidate, candidate count for the entity, source indicator (S2 vs S3), embedding cosine + rank (once Tanuj delivers).

**Competition features** (many-to-one structure): for each candidate, its best score across every entity that proposed it, margin to runner-up, whether current entity is its argmax. Pass 1 uses `prior_score`; pass 2 uses model probabilities (small stacking step, Gate 3).

### D3 — Evaluation harness

`s7_evaluate.py`: overall macro F0.5 with precision/recall separately; broken down by country; broken down by true-match-count bucket (0, 1, 2-3, 4-5, 6+); singleton accuracy; blocking recall ceiling. Write every report to `artifacts/reports/<git-sha>.json`.

### D4 — Error analysis

From Gate 2: dump 100 worst false positives and false negatives as readable side-by-side text with feature values. Post the patterns found — fills section 5 of the methodology doc.

---

## Timeline

~52 hours remain. A valid submission must exist by Gate 1, Friday night.

| Gate | By when | What must exist |
| --- | --- | --- |
| G0 | Fri 22:30 | Repo, schemas, stubs, smoke sample, validation split, country answer |
| G1 | Sat 04:00 | End-to-end run producing a validated submission. First upload. |
| G2 | Sat 16:00 | Full blocking with measured recall, LightGBM v1, calibration, transfer test |
| G3 | Sun 04:00 | Embedding channel, competition features, expected-F0.5 selection live |
| G4 | Sun 14:00 | Feature freeze. Final training run, full test inference. |
| G5 | Sun 20:00 | Zip packaged, methodology written, validator passing. Buffer. |

### Per-cycle ownership

| | Nidhi | Tanuj | Parth | Krrish |
| --- | --- | --- | --- | --- |
| To G0 | Repo, ingest, split, stubs, country check | Environment, Kaggle dataset upload | Acceptance-test file | `metric.py` + unit test |
| To G1 | Exact-key + rare-token channels; naive assembly | Trivial model on 10 features | Normalisation v1 on smoke sample | `featurise` v1, 20 features |
| To G2 | TF-IDF channels full scale; recall report | Training set, LightGBM v1, calibration, transfer test | Full normalisation, lexicons, mined table | Full feature set, evaluation report |
| To G3 | Uniqueness constraint, expected-F0.5 search | Embeddings, ANN index, cosine feature | Mined lexicon v2 from error analysis | Competition features, error dumps |
| To G4 | Final inference, threshold tuning, uploads | Final training, ensemble if time | Methodology sections 2-3 | Methodology sections 5-6 |
| To G5 | Package zip, run validator | requirements.txt, reproduction README | Lexicon documentation | Final report, charts |

Nobody waits on a real implementation — stubs exist from Gate 0. Merge to `main` only through a green smoke run.

---

## Submission strategy

15 uploads total (5/day × 3 days).

**Submit an all-empty prediction first.** Because k=0 scores 1.0 for empty and 0.0 otherwise, the returned score is exactly the singleton rate of the public test subset — confirms format acceptance and gives a measured prior for P(k=0) on real test distribution.

**Rules for the rest:** run the validator before every upload without exception; never upload without scoring locally first; don't chase the public leaderboard over local validation; keep 2 of Sunday's 5 uploads for after the final training run; change one thing at a time past Gate 2.

Keep `submissions.md`: # | Time | Git SHA | Change | Local F0.5 | Public LB.

---

## Risks

| Risk | How you notice | What to do |
| --- | --- | --- |
| Blocking recall far below 97% | Nidhi's G2 recall report | Add channels and raise the cap before touching the model |
| Feature matrix won't fit on disk | 25GB write fails mid-run | Fuse featurise+score, shard at 5M pairs, keep only probabilities |
| Kaggle session dies mid-embedding | Notebook stops, no output | Checkpoint every 500k records, resume from last shard |
| France collapses the score | Tanuj's transfer test at G2 | Drop country features, tighten monotonic constraints |
| Indic text arrives as question marks | Parth's acceptance test fails | `PYTHONUTF8=1`, explicit `encoding="utf-8"` everywhere |
| Final inference outruns the clock | Unknown runtime at G4 | Time the full test inference at Gate 3 |
| A track stalls | No merge in six hours | Gate-0 stub keeps pipeline running; redistribute the piece |
| Submission rejected on format | Validator output | Run `validate_submission.py` before every upload |
| Local/leaderboard diverge | Gap column in submissions.md | Suspect a bug before overfitting |
| Disk fills mid-run | Write errors on artifacts/ | Delete superseded stage outputs at each gate |

---

## Final package

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/business_entity_resolution/
│   ├── src/
│   ├── README.md
│   └── requirements.txt
└── Documentation_template.md
```

Methodology document must include: (1) the metric derivation showing F0.5 = 1.25c/(0.25k+m) and the resulting expected-value selection search; (2) the many-to-one discovery and how the uniqueness constraint + competition features exploit it; (3) the mined transliteration lexicon, with sample mappings and quantified recall gain; (4) the blocking recall analysis with per-channel drop-one contribution.

### Final checklist

- [ ] `validate_submission.py` prints PASS on the exact files in the zip
- [ ] `matching_results.tsv` has 1,732,545 lines including header
- [ ] Every ID in results also appears in that entity's `candidate_pairs.tsv` row
- [ ] No `S1-` IDs in any match list, no IDs absent from test files
- [ ] Singleton rows are genuinely empty, not `nan`
- [ ] Both files tab-separated, UTF-8
- [ ] `requirements.txt` pins every version; model licence named in write-up
- [ ] `submissions.md` version history complete
- [ ] Zip structure matches exactly, `Documentation_template.md` still named that

---

## Decisions log

### 2026-09-26 — Candidate cap set to 30 (provisional)

Following the organisers' update that candidate_pairs.tsv size is scored
independently of the leaderboard, ran a full cap sweep (K=10,15,20,25,30,
40,50,60,uncapped) on the smoke validation set. Recall gain per additional
candidate flattens sharply past K=30 (30→40 gains 0.16pts, 50→60 gains
only 0.04pts). Chose K=30: 0.9878 recall (India 0.9805, US 0.9926) vs
0.9908 at K=60, a 0.30-point cost for a 45.5% reduction in candidate
pairs (2,752,283 to 1,499,998 on smoke train).

Per-channel finding: pairs found by multiple channels are almost never
cut by the cap (>99% survive even at K=10) — nearly all recall loss at
low K comes from pairs only ONE channel found, and rare_token's exclusive
finds are hit hardest (only 48% survive at K=20 vs 85-88% for the two
TF-IDF channels), because the ranking score sums 1/rank across channels
and a single-channel find has nothing to add to its rank.

Caveats, must revisit before finalising: (1) smoke's per-country shards
are 93k-140k records vs 4-5M at full scale, so full-scale recall at any
given K will sit lower than measured here — this cap is not validated at
scale yet. (2) embed_ann contributed 0 pairs (not yet wired in) — its
addition will change the union and may shift the right K. (3) France has
no ground truth, so this analysis says nothing about French recall.

Full sweep data: artifacts/smoke/reports/cap_sweep.json (gitignored,
regenerate with src/sweep_candidate_cap.py --smoke).

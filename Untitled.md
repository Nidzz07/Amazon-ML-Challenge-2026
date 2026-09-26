# Amazon ML Challenge 2026 — Business Entity Resolution

Sep 25, 2026 · @Nidhi Dhyani

The scored quantity is macro-averaged F₀.₅ over 1,732,544 test Source-1 entities, and it collapses to one expression per entity: **F = 1.25c / (0.25k + m)**, where c is correct predictions, k is true matches and m is how many you predicted. Every decision below exists to maximise that expression before Sep 27, 2026, 23:59 IST.

Team of four working in parallel: Nidhi (candidate generation + assembly), Tanuj (matching model), Parth (normalisation + transliteration), Krrish (features + evaluation).

## Scoreboard math

Substituting P = c/m and R = c/k into Fβ with β = 0.5 cancels almost everything:

```latex
F_{0.5} = \frac{1.25 \cdot P \cdot R}{0.25P + R} = \frac{1.25\,c}{0.25\,k + m}
```

c = matches you got right, k = true matches for that entity, m = how many you predicted. Checked against the organisers' own worked example: predicting 3 when 2 of them are right and k = 2 gives 1.25·2/(0.5+3) = 0.714, exactly the number printed in the problem statement. The metric is linear in c and has a constant 0.25k offset in the denominator, which produces three consequences that most teams will miss.

**1. The optimal threshold is not 0.5, and it is not constant.** Ask whether to add one more candidate that is correct with probability q. Expected score after adding is 1.25(c+q)/(0.25k+m+1), before adding it is 1.25c/(0.25k+m). Adding is worth it precisely when

```latex
q > \frac{c}{0.25\,k + m}
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

**3. The selection layer should be a search, not a threshold.** Sort an entity's candidates by calibrated probability descending, then evaluate expected F₀.₅ for every prefix length m = 0 to n and take the best. Plug in ĉ = Σᵢ≤ₘ qᵢ and k̂ = Σ over all candidates qᵢ; that approximation is close enough at 1.7M entities and costs microseconds each. Krrish calibrates the residual bias on the validation split.

This selection layer is Nidhi's, it is independent of whichever model produces q, and it is the single highest-leverage component in the pipeline.

## Data profile

Measured directly from the files, not assumed. Naive comparison is 1.73M × 9.97M ≈ 1.7×10¹³ pairs, so blocking is not a stage, it is the problem.

| File | Rows | Size |
| --- | --- | --- |
| train\_source1.tsv | 2,206,821 | 210 MB |
| train\_source2.tsv | 5,034,616 | 489 MB |
| train\_source3.tsv | 5,285,603 | 504 MB |
| train\_ground\_truth.tsv | 2,206,821 | 127 MB |
| test\_source1.tsv | 1,732,544 | 175 MB |
| test\_source2.tsv | 4,887,273 | 509 MB |
| test\_source3.tsv | 5,082,316 | 506 MB |

**Each Source-2 / Source-3 record belongs to at most one Source-1 entity.** All 7,638,365 matched IDs in the ground truth are distinct — zero reuse across the 2.2M rows. The true structure is many-to-one, which licenses a global uniqueness constraint at assembly and a set of competition features during scoring. This is the strongest structural prior in the dataset and it is not stated anywhere in the problem statement.

**Label shape.** Mean 3.46 matches per entity, mode 3, maximum 11. 5.58% are singletons (123,247 rows). Of the matched IDs, 3,693,619 come from Source 2 and 3,944,746 from Source 3. Roughly 26% of all Source-2/3 records match nothing at all and exist purely as distractors.

**Country.** Training is US (1,323,633 S1) and India (883,188 S1). Test adds France: 259,452 S1 entities, 15% of the test set, with no training data whatsoever. Every sampled positive pair was same-country, but **verify this before relying on it** — it is Hour-0 task 1, and if even 0.1% of true pairs cross countries, country becomes a soft feature instead of a hard partition.

**Missing addresses.** 168,967 train-S2 and 175,916 train-S3 rows have an empty address; 129,408 and 136,098 in test. Those records can only be matched on name.

### Noise catalogue, from real positive pairs

Every pattern below was read off actual matched groups in the training data.

- **Cross-script transliteration.** `Raj Investments LLP` matches `ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி`. `Ss Food Private Limited` matches `एसएस फूड प्राइवेट लिमिटेड`. States appear as `तमिलनाडु`, `ಕನಾಱ್ಟಕ`, `TN`, `Tamil Nadu`. Latin-only string similarity scores these near zero. Devanagari, Tamil and Kannada all appear; assume more scripts exist.
- **Name-blind matches.** `Maure Williams Colombier Inc` at 85 Wayne Avenue matches `Dréxkor` at `85 Wanye Avenue` — no name overlap at all, address only. A name-only blocker has a hard recall ceiling.
- **Address-blind matches.** The same entity also matches two records with completely empty addresses. You need both channels, independently.
- **Injected accents.** `Enterprises` → `Énterprises`, `Boral` → `Bóral`. Strip accents for matching but keep the original.
- **Character-level typos.** `Enterpires`, `Etrepndiels`, `Wanye`, `AKON`, `Wilblims`, `Ponr`. Edit distance still works; token equality does not.
- **Abbreviation drift.** `St` / `Street` / `SAINT`, `Rd` / `Road`, `Ave` / `Avenue`, `IL` / `Illinois`, `Pvt` / `Private`, `Ltd` / `Limited`.
- **Component reordering.** `630 45th Terrace, Kansas City, MO` appears as `KANSAS CITY, MO, 630 45ND TERRACE, null`. Token-set similarity survives this; sequence similarity does not.
- **Literal `null` tokens** embedded mid-address, plus `PO BOX` insertions and house-number mutations (`630 45th` → `45ND`, `1056` → `1056c`).
- **Domain names as business names.** `maurewilliamscolombier.com` matches `Maure Williams Colombier Inc`. Strip the TLD and split the stem.
- **Legal-suffix drift.** The suffix is present, absent, or moved in roughly a third of positives. France brings `SARL`, `SAS`, `SASU`, `SCI`, `EURL`, none of which appear in training.

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

**Assembly enforces the many-to-one constraint** discovered in the data profile: each Source-2/3 record may be claimed by only one Source-1 entity. Resolve conflicts greedily by descending probability, then run the expected-F₀.₅ prefix search per entity.

## Repo and file contracts

The repo mirrors the required zip structure from day one, so packaging at the end is a copy rather than a reorganisation.

```markdown
amazon-ml-2026/
├── code/business_entity_resolution/
│   ├── src/
│   │   ├── config.py          # all paths and knobs, no hard-coded literals elsewhere
│   │   ├── s0_ingest.py … s7_evaluate.py
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

**`norm_{split}_{src}.parquet`** — `entity_id` str, `name_norm` str, `name_roman` str, `name_tokens` list\[str\], `name_acronym` str, `addr_norm` str, `addr_roman` str, `addr_tokens` list\[str\], `street_num` str, `city_norm` str, `state_canon` str, `postcode` str, `country` str, `has_addr` bool, `script` uint8.

**`candidates_{split}.parquet`** — `source1_entity_id` str, `candidate_entity_id` str, `channels` uint8 bitmask, `n_channels` uint8, `best_rank` uint16, `prior_score` float32.

**`features_{split}.parquet`** — the two ID columns, then `f000 … fNNN` float32 in the exact order of `features.FEATURE_NAMES`, plus `label` uint8 on train splits only.

`features.FEATURE_NAMES` is the single source of truth for feature order and carries a `FEATURE_VERSION` integer. Tanuj's model file records the version it was trained against and `s5_score.py` refuses to run on a mismatch. That one guard prevents the most common way a four-person ML pipeline silently produces garbage.

### Git and artifact discipline

- Branch per track: `track/blocking`, `track/model`, `track/normalise`, `track/features`. Merge to `main` only when the stage runs end to end on the 50k smoke sample.
- `.gitignore` covers `data/`, `artifacts/`, `output/`, `*.parquet`, `*.txt` model files. **Never commit an artifact** — a 2 GB push will cost you an hour you do not have.
- Google Drive holds artifacts only: `/amazon-ml-2026/artifacts/<stage>/<date-hhmm>/`. Name every upload with the git commit SHA that produced it. An artifact whose producing commit is unknown is worthless.
- Everything is seeded and deterministic. `config.py` sets `SEED = 42` and every stage reads it. Two people running the same commit on the same input must get byte-identical output, or debugging becomes impossible.

### The smoke sample

Before anything else, Nidhi cuts a 50,000-entity slice — `smoke_*` files with the full schema — that runs the entire pipeline in under two minutes on the weakest laptop. Every track develops against it. Full-scale runs happen only at gates. Without this, Parth and Krrish will spend the hackathon waiting on I/O instead of writing code.

## System setup

| Machine | Spec | Role |
| --- | --- | --- |
| Tanuj | Ryzen 7, 24 GB DDR5, RTX 4050 6 GB | Heavy compute box. All full-scale blocking runs and all GPU embedding jobs land here. |
| Parth | i5-13th gen, 16 GB, RTX 4050 6 GB | Normalisation at full scale; second GPU for embedding shards. |
| Krrish | i5-13th gen, 16 GB, RTX 3050 4 GB | Feature development and the evaluation harness. |
| Nidhi | Ryzen 5 5500U, 15.3 GB usable, integrated Radeon, 136 GB free | Integrator: design, assembly, submission. No CUDA. |

**One problem worth naming.** You own blocking, which is the most compute-hungry stage, on a low-TDP mobile chip with no usable GPU. Do not fight this. Design and debug blocking on the smoke sample locally, then run the full-scale pass on Tanuj's 24 GB box or in a Kaggle CPU notebook, which gives 30 GB of RAM — twice your laptop. Your machine's job is integration and the two output files, and that work is light.

### Environment, identical on all four machines

Python **3.11** — not 3.12 or 3.13, several wheels still lag. One virtual environment per person, `requirements.txt` pinned and committed at Gate 0.

```markdown
polars pyarrow pandas numpy scipy scikit-learn
lightgbm rapidfuzz jellyfish python-Levenshtein
indic-transliteration aksharamukha regex unidecode
faiss-cpu sparse-dot-topn
sentence-transformers transformers
tqdm orjson pytest
```

GPU machines add PyTorch separately: `pip install torch --index-url https://download.pytorch.org/whl/cu121`.

### Four Windows problems that will cost you hours if unhandled

1. **Encoding.** Windows Python defaults to cp1252, which silently mangles Devanagari, Tamil and Kannada. Set `PYTHONUTF8=1` as a user environment variable on every machine, and pass `encoding="utf-8"` explicitly on every file open regardless. If your normalised names come back as question marks, this is why.
2. **OneDrive.** Do not put `artifacts/` or `data/` under `Documents` or `Desktop` if OneDrive syncs them — it will try to upload 40 GB. Keep the repo somewhere like `C:\ml2026\`.
3. **Long paths.** Enable them: `Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -Value 1` in an admin PowerShell, then reboot.
4. **Tabs.** Read every file with `sep="\t"` and write with `sep="\t"`. The addresses contain commas; the organisers warn that reading without the explicit separator produces one column and fails silently.

### Disk budget, and the one thing that will not fit

Parquet intermediates run roughly 40–50 GB. That fits everywhere. What does **not** fit is a materialised feature matrix: 1.73M test entities × 60 candidates × \~60 float32 features is about 25 GB for test alone, and more for train.

So **S3 and S5 are fused and sharded at inference**: compute features for a shard of roughly 5M pairs, score it immediately, write only `(source1_entity_id, candidate_entity_id, prob)`, discard the features. Only training data is materialised, and only after negative subsampling — keep all 7.6M positives plus hard negatives at 2:1, on a 40% entity subsample, which lands near 15M rows and 4 GB. Tanuj sets that ratio at Gate 2.

### Kaggle, for the GPU stages

Upload the dataset once as a **private Kaggle Dataset** rather than re-uploading per session; it then mounts read-only at `/kaggle/input/` in every notebook. Free tier gives a T4, sessions up to 9 hours, roughly 30 GPU-hours per week per account — four Google accounts is ample. Write to `/kaggle/working/` (20 GB) and push results out as a new dataset version so the next session can mount them.

Embedding 12M records with a small multilingual encoder runs near 2,000 records/second on a T4, so a full pass is under two hours. **Checkpoint every 500k records** — free sessions die without warning.

### Licence compliance

The rules require the final model to be MIT or Apache-2.0 and under 8B parameters. Compliant choices: LaBSE (Apache-2.0, 471M), multilingual-e5-small or base (MIT, 118M/278M), BGE-M3 (MIT, 568M). LightGBM is MIT. Record the chosen model and its licence in the methodology document — the top teams' packages are reviewed before final rankings.

Also: **no external lookups of any kind.** No geocoding, no business registries, no entity-resolution APIs, no scraped reference data. Detection means immediate disqualification. Every transformation must derive from the provided files or from offline libraries and hand-written tables.

## Track A — Nidhi: candidate generation and assembly

You own the recall ceiling and the two files that get scored. Nobody else can unblock you and everybody else is blocked by you at Gate 0, so the first three hours are the most valuable three hours anyone on the team will spend.

### A0 — Hour 0 to 3, blocking everyone else

1. **Verify the country assumption.** Join ground truth to the source files and count matched pairs whose country differs from their Source-1 entity's. If it is exactly zero, country becomes a hard partition and the problem shrinks threefold. If it is non-zero, country becomes a feature and blocking runs cross-country with a penalty. Post the number in the group chat — three people are waiting on it.
2. **Repo bootstrap.** Folder structure, `config.py`, `.gitignore`, `requirements.txt`, four branches, push.
3. **`s0_ingest.py`.** TSV to Parquet with the fixed schema. Verify round-trip row counts against the numbers in the data profile above.
4. **Validation split.** 400,000 Source-1 training entities held out, seeded, written as an ID list. Critical detail: the Source-2/3 pool stays **complete** for validation. Records belonging to training entities act as distractors, which is exactly the situation at test time. Shrinking the pool would flatter every score you measure.
5. **Smoke sample.** 50,000 entities, full schema, whole pipeline under two minutes.
6. **Stub every stage.** `s1` through `s7` as runnable no-ops that read and write the right schemas. This lets the other three start immediately against a working skeleton instead of waiting for real code.

Gate 0 closes when all four branches exist, the smoke sample runs end to end through the stubs, and the country number is posted.

### A1 — Blocking channels

Five channels, each a module under `src/blocking/`, each emitting `(source1_entity_id, candidate_entity_id, channel_id, channel_rank, channel_score)`.

- **`name_tfidf`** — character 3–4-gram TF-IDF over `name_roman`, top 20 per entity. Build the index per country shard. Use `sparse_dot_topn` with query chunks of 20,000 rows. If the India shard (810k queries against 4.7M records) runs past 40 minutes, switch to `TruncatedSVD(256)` on the TF-IDF matrix and a `faiss` IVFFlat index — far faster, slight recall cost, measure both.
- **`addr_tfidf`** — identical, over `addr_roman`, top 20. Skip records where `has_addr` is false.
- **`exact_key`** — three key families: `(street_num, city_norm)`, `(postcode, street_num)`, `(name_acronym, city_norm)`. Hash join, no ranking. Drop any key whose bucket exceeds 500 records; those are noise, not signal.
- **`rare_token`** — inverted index over name and address tokens with document frequency between 2 and 5,000. For each entity take its three rarest tokens and union their postings. This is what catches the reordered-address cases.
- **`embed_ann`** — Tanuj's embeddings, `faiss` top 20. Plugs in at Gate 3; the pipeline must work without it.

**Union and cap.** Deduplicate on the pair, set the `channels` bitmask and `n_channels`, compute a cheap `prior_score` as a rank-weighted blend across channels, and keep the top 60 per entity. The capped set is what goes into `candidate_pairs.tsv` — the rules define that file as the last filtering stage before the model, so the cap must happen here, not later.

### A2 — The measurement that matters most

On the validation split, report **recall of true pairs present in the candidate set**, broken down by country and by channel, plus a drop-one analysis showing each channel's marginal contribution. Target recall is 97% or better at a median of 60 candidates per entity.

This number is your recall ceiling. No model, however good, can exceed it. Measure it before Tanuj trains anything, because if it reads 82% the right response is another blocking channel, not a better classifier.

### A3 — Assembly

Two steps, in order, in `assemble.py`.

1. **Uniqueness constraint.** Each Source-2/3 record belongs to at most one Source-1 entity. Group scored pairs by `candidate_entity_id` and keep only the highest-probability claim. Build a soft variant too — strip a claim only when the margin over the runner-up exceeds a tuned threshold — and pick between them on validation. The hard version is usually better under F₀.₅ because it is purely precision-buying.
2. **Expected-F₀.₅ prefix search.** Per entity, sort candidates by calibrated probability descending, compute expected F₀.₅ for every prefix length from 0 to n using ĉ = Σᵢ≤ₘ qᵢ and k̂ = Σ over all qᵢ, take the argmax. m = 0 is the singleton prediction and must be a genuine option in the search, not a special case bolted on afterwards.

### A4 — Output and validation

Write both TSVs with `sep="\t"`, `encoding="utf-8"`, no quoting, no index. Then check, in code, before every single upload:

- exactly 1,732,544 rows plus header in each file, one per test Source-1 entity, no duplicates
- `matched_entity_ids` is the empty string for singletons, never `NaN`, never the literal `nan`
- no duplicate IDs inside any list
- every ID in `matching_results.tsv` also appears in that entity's `candidate_pairs.tsv` row
- every ID exists in the test source files, and no `S1-` ID appears in any list

Then run the organisers' own validator from the `student_resource/` directory and require `PASS` before uploading. It costs thirty seconds and a failed submission costs one of five daily attempts.

### Kickoff prompt for your Claude Code session

```markdown
Set up the repo skeleton for the Amazon ML Challenge entity-resolution
pipeline, following the roadmap's stage and schema contract.

1. Create the folder structure under C:\ml2026\amazon-ml-2026 exactly as
   specified, with code/business_entity_resolution/src/ holding config.py
   and stage scripts s0 through s7.
2. config.py: paths to the dataset, SEED=42, per-stage knobs, no literals
   anywhere else.
3. s0_ingest.py: read the seven TSVs with sep="\t", encoding="utf-8",
   dtype=str, write Parquet with the records_{split}_{src} schema.
   Assert the row counts match: train 2206821/5034616/5285603,
   test 1732544/4887273/5082316.
4. A script that answers one question: across train_ground_truth, how many
   matched pairs have a country different from their Source-1 entity's?
   Print the count, the percentage, and the top cross-country combinations.
   Stream it - do not load all 12M records into memory at once.
5. Stub s1 through s7 as runnable no-ops with correct input/output schemas.
6. A 50k-entity smoke sample generator preserving the full schema.

I have 15GB RAM. Use polars lazy scans, not pandas, for anything touching
the full files.
```

## Track B — Tanuj: the matching model

You produce one thing: a **calibrated** probability that a candidate pair is a true match. Calibration is not optional polish here — Nidhi's selection layer compares probabilities against thresholds derived from the metric, so a model that ranks perfectly but is systematically over-confident will destroy the score. Rank quality and calibration are separate deliverables and both are yours.

### B1 — Training data construction

The way you sample negatives matters more than the model.

- Keep **all 7.6M positives**.
- Draw negatives **only from the blocking output** — the highest-ranked non-matching candidates. Randomly sampled negatives are trivially separable, the model learns a boundary that does not exist at inference time, and validation looks wonderful while the leaderboard does not move.
- Ratio 2:1 negative to positive, on a 40% entity subsample. That lands near 15M rows and about 4 GB.
- **Split by entity, never by pair.** An entity's pairs all go to train or all to validation. Otherwise the model memorises entities and every number you report is fiction.

### B2 — The model

LightGBM, binary objective, on Krrish's feature matrix. Start with 500 trees, learning rate 0.05, 63 leaves, and tune only after the pipeline is closed end to end.

Two choices that matter more than hyperparameters:

**Monotonic constraints.** Every similarity feature should be constrained monotonically increasing — more name overlap can never reduce match probability. Pass `monotone_constraints` for those columns. This costs a little training fit and buys a lot of generalisation, which is exactly the trade you want for a test set containing a country you have never seen.

**Keep country out of the main model.** Train the primary model without any country feature. Train a second variant with it. Compare them on the transfer test below and let the number decide — but the default is out, because France has no training data and a country-conditioned model has no idea what to do with it.

### B3 — Calibration

Fit isotonic regression on a held-out slice that was used for neither tree-fitting nor threshold tuning. Report a reliability curve and Brier score, not just AUC. Hand Nidhi a `calibrator.pkl` and confirm that predicted probabilities near 0.7 are correct about 70% of the time — that is the band where the selection layer makes most of its decisions, so accuracy there matters more than anywhere else.

### B4 — The France proxy

You cannot validate on France. You can approximate it: **train on US-only entities and evaluate on India-only entities**, then the reverse. The drop between in-country and cross-country performance is your best estimate of what happens to 15% of the test set.

Run this at Gate 2 and report both numbers. If the drop is severe, the fix is more country-agnostic features and tighter monotonic constraints, and it is better to learn that on Saturday than from the private leaderboard on Sunday night.

### B5 — The embedding channel

This is what makes cross-script matching work, and it is yours because you have the strongest GPU.

- Model: **LaBSE** (Apache-2.0, 471M) or **multilingual-e5-base** (MIT, 278M). Both are licence-compliant and both handle Devanagari, Tamil, Kannada and French natively. Start with e5-base for speed; e5 needs the `"query: "` prefix convention, LaBSE does not.
- Embed `name_roman` **and** the raw `business_name` — the raw form carries the native script, which is the entire point. Encode name and address as separate vectors; some positives share only a name and others share only an address, so one merged vector blurs both signals.
- Normalise vectors, build a `faiss` index per country shard, retrieve top 20. India is the biggest shard at 4.7M × 384 dims, about 7 GB in fp32 — fine in a Kaggle CPU notebook with 30 GB.
- Deliver two things: the ANN pairs for Nidhi's fifth blocking channel, and the cosine similarity as a feature for Krrish. The same embeddings serve both.
- **Checkpoint every 500k records.** Kaggle sessions die silently.

### B6 — Only if Gate 3 lands early

A cross-encoder over the top 5 candidates per entity, fine-tuned on positives and hard negatives. Highest ceiling of anything in this plan and the easiest way to run out of time. Do not start it before the full pipeline has produced a validated submission.

### Kickoff prompt for your agent

```markdown
I am building the matching model for a large-scale entity-resolution task.
Input is a Parquet feature matrix: two ID columns, ~60 float32 feature
columns in a fixed order, and a uint8 label. Output must be a calibrated
P(match) per pair.

Build:
1. build_training_set.py - takes blocking candidates plus ground truth,
   keeps all positives, samples hard negatives from top-ranked non-matches
   at 2:1, subsamples to 40% of entities, splits by ENTITY not by pair.
2. train.py - LightGBM binary, monotonic constraints on every similarity
   feature (read the constraint direction from features.FEATURE_NAMES
   metadata), early stopping on the entity-grouped validation split.
3. calibrate.py - isotonic regression on a third disjoint slice; emit a
   reliability curve and Brier score.
4. transfer_test.py - train on US-only entities, evaluate on India-only,
   and the reverse. Report the degradation. This estimates performance on
   France, which appears only in the test set.
5. score.py - load model plus calibrator, verify FEATURE_VERSION matches
   what the model was trained on, refuse to run on mismatch, score in
   shards, write only (source1_entity_id, candidate_entity_id, prob).

24GB RAM, RTX 4050. Never materialise the full inference feature matrix -
it is 25GB. Shard at ~5M pairs, featurise and score, discard, repeat.
```

## Track C — Parth: normalisation and transliteration

Your track is the one most likely to separate this team from the field. Roughly 40% of the data is Indian and a large share of its positive pairs are written in different scripts — `Raj Investments LLP` against `ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி`. Every string metric in the world scores that pair at zero. Teams that skip this lose most of the Indian recall and never find out why.

Everything you write is a **pure function**: string in, string out, no I/O, no global state, fully unit-testable. That is what lets you work at full speed while the rest of the pipeline is still being built.

### C1 — The normalisation pipeline

In order: NFKC Unicode normalisation, lowercase, strip accents into a separate field (keep the accented original), collapse whitespace and punctuation, remove junk tokens (`null`, `n/a`, `-`, `--`, `<<`, `nil`, empty parentheses).

**Always emit both the normalised and the original.** Krrish needs both — the accent stripping that helps `Énterprises` match `Enterprises` also destroys a real signal elsewhere, and only the feature layer can decide.

### C2 — The lexicons

Hand-written tables under `src/lexicon/`, each a plain mapping file:

- **Legal suffixes** by country. US: `inc`, `llc`, `corp`, `corporation`, `co`, `ltd`, `limited`, `plc`, `lp`, `llp`. India: `pvt`, `private`, `ltd`, `limited`, `llp`, `and company`, `& co`. France: `sarl`, `sas`, `sasu`, `sa`, `sci`, `eurl`, `snc`. Emit a suffix-stripped name **and** a flag recording what was removed — two companies differing only by suffix are usually the same entity, but not always.
- **Street abbreviations**: `st`/`street`/`saint` (genuinely ambiguous — `SAINT` appears in the data as an expansion of `St`), `rd`/`road`, `ave`/`avenue`, `blvd`, `ln`, `dr`, `ct`, `sq`, `hwy`, `ste`/`suite`, `apt`, `fl`/`floor`, `po box`.
- **State and region canonicalisation**, three-way: full name, official code, and native script. US needs all 50 states plus DC. India needs 28 states and 8 union territories in Devanagari, Tamil, Kannada, Telugu, Bengali, Gujarati, Malayalam, Marathi, Punjabi and Odia — the data already contains `तमिलनाडु`, `ಕನಾಱ್ಟಕ`, `TN` and `Tamil Nadu` for the same state. France has no training data, so handle its regions generically.
- **Business-word glossary**: the recurring nouns — `food`, `investments`, `properties`, `marketing`, `finance`, `infratech`, `enterprises` — in each script.

### C3 — Script detection and romanisation

Detect script per field by Unicode block and emit it as a flag. Romanise every non-Latin string with `indic-transliteration` or `aksharamukha`, both MIT-licensed and fully offline. Write the result to `name_roman` and `addr_roman`.

Romanisation alone is not enough: `राम` romanises to `rāma`, while Source 1 holds `Ram`. Fold diacritics, collapse doubled vowels, and let the fuzzy matcher close the remaining gap. Never expect exact equality after transliteration.

### C4 — The mined lexicon, and why it beats generic romanisation

This is the part worth writing up in the methodology document.

The ground truth hands you 2.2M groups of records that are known to be the same entity. Within a group you often have the same business name in Latin and in Devanagari. Align them:

1. For every training group, tokenise the Source-1 name and each matched name.
2. Where one side is native-script and the other Latin, and the token counts are close, record co-occurrence counts between token positions.
3. Score pairs by pointwise mutual information and keep those with support of 20 or more.
4. You now have a data-derived mapping: `प्राइवेट → private`, `एलएलपी → llp`, `फूड → food`.

Apply this lookup **before** generic romanisation and fall back to the library only for unknown tokens. The same technique mines Latin abbreviation pairs (`corp` ↔ `corporation`) without anyone hand-writing them.

This uses only the provided training data, so it is fully within the fair-play rules — state that explicitly in the write-up, because it is the kind of thing a reviewer will look at twice.

### C5 — Address parsing

From each address produce `street_num`, `city_norm`, `state_canon`, `postcode`, and the token list. Be forgiving: components arrive reordered, the house number mutates (`630 45th` becomes `45ND`, `1056` becomes `1056c`), and landmark references like `Near SBI ATM` appear mid-string. Extract the leading digit run as `street_num` and keep the full token too.

### Acceptance test — build this first

Before writing any normalisation code, write the test. Take 30 real positive pairs from the noise catalogue in the data profile above and assert that after normalisation each pair scores above a similarity floor. That file is your definition of done, it tells you instantly when a change helps or hurts, and it is the only way to work on this track without a full pipeline run.

### Kickoff prompt for your agent

```markdown
I am writing the normalisation layer for a multilingual business entity
resolution task. US, Indian and French business names and addresses, where
the same entity appears in Latin, Devanagari, Tamil and Kannada scripts.

Build, in this order:
1. tests/test_normalise.py FIRST - 30 known-positive pairs (I will paste
   them), asserting each pair exceeds a similarity floor after
   normalisation. Tests before implementation.
2. normalise.py - pure functions, no I/O. NFKC, lowercase, accent handling
   that keeps both forms, junk-token removal, punctuation collapse.
3. lexicon/ - legal suffixes per country, street abbreviations, and a
   three-way state map (full name, code, native script) for all Indian
   states and union territories and all US states.
4. translit.py - Unicode-block script detection plus romanisation via
   indic-transliteration or aksharamukha. Offline only, MIT-licensed only.
5. mine_lexicon.py - the important one. Read train_ground_truth plus the
   source files. Within each matched group, align native-script names to
   Latin names, count token co-occurrences, score by PMI, keep support>=20,
   emit a token mapping table. Apply it before generic romanisation.

Everything must be a pure function and vectorised or fast enough for 12.5M
records. No network calls anywhere - external data lookup means
disqualification.
```

## Track D — Krrish: features and evaluation

Two deliverables. The feature library is what the model actually learns from. The evaluation harness is how the whole team knows whether anything is working — and until it exists, nobody can tell an improvement from a regression.

**Write `metric.py` first.** It is thirty lines and it unblocks all four of you.

### D1 — metric.py, the first hour

Implement macro-averaged F₀.₅ exactly as specified, using the closed form `F = 1.25c / (0.25k + m)`. Rules that are easy to get wrong:

- An entity with no true matches scores 1.0 for an empty prediction and 0.0 for any non-empty one.
- Singletons are included in the average, not skipped.
- A missing entity in the prediction file is an error, not a zero — raise, do not silently score it.
- Average over entities, never over pairs.

**Verify against the organisers' worked example before trusting it**: prediction `[S2-00047, S2-00193, S3-00812]`, truth `[S2-00047, S3-00812]` must return exactly 0.714. Commit that as a unit test. If this file is wrong, every number the team produces for the next two days is wrong.

### D2 — The feature library

One function: `featurise(s1_rows, cand_rows, context) -> np.ndarray` of float32, columns in the exact order of `FEATURE_NAMES`. Each entry in `FEATURE_NAMES` carries a name and a monotonic direction, which Tanuj reads to set his LightGBM constraints.

**Name features**, computed on both `name_norm` and `name_roman`: `rapidfuzz` ratio, partial ratio, token-sort ratio, token-set ratio, Jaro-Winkler; character 3-gram and 4-gram Jaccard; token Jaccard and containment; IDF-weighted token cosine; acronym match; first-token match; common-prefix ratio; length ratio; suffix-stripped ratio plus a suffix-agreement flag; digit-token equality; domain-stem match for the `.com` cases; a script-pair flag.

**Address features**: the same similarity family over `addr_norm` and `addr_roman`, plus exact-match indicators for `street_num`, `postcode`, `state_canon` and `city_norm`, numeric-token Jaccard, and `has_addr` flags for both sides. Note that token-set similarity survives the component reordering seen in the data while sequence similarity does not — include both so the model can learn when each applies.

**Context features**, which are where the real gains hide. These describe the candidate's position among its competitors rather than the pair in isolation:

- `n_channels` and one flag per blocking channel that retrieved the pair. A pair found by four channels is far more likely to be a true match than one found by one.
- Rank within the entity by each channel, and the reciprocal rank.
- Margin between this candidate's score and the entity's best candidate.
- Number of candidates the entity has at all.
- Source indicator, S2 against S3.
- Embedding cosine and embedding rank, once Tanuj delivers them.

**Competition features**, exploiting the many-to-one structure. For each candidate record, compute its best score across every Source-1 entity that proposed it, the margin to its runner-up, and whether the current entity is its argmax. A record that three entities all want is evidence against all three; a record only one entity wants is evidence for that one.

These need scores to exist, so pass one uses `prior_score` from blocking and pass two uses the model's own probabilities — a small stacking step that lands at Gate 3.

### D3 — The evaluation harness

`s7_evaluate.py` takes a prediction file and a ground-truth file and emits a JSON report plus a readable summary:

- Overall macro F₀.₅, with precision and recall reported separately so the team can see which side is failing.
- **Broken down by country** — US against India is the only early warning available for France.
- Broken down by true match count: singletons, 1, 2–3, 4–5, 6 or more. Different failure modes live in different buckets.
- Singleton accuracy specifically: how often an empty prediction was correct, and how much score was lost by predicting on true singletons.
- Blocking recall ceiling, so it is always visible how much of the loss is unreachable by the model.

Every report is written to `artifacts/reports/<git-sha>.json`. When someone asks on Sunday morning which configuration scored best, the answer must be a lookup, not an argument.

### D4 — Error analysis, from Gate 2 onwards

Dump the 100 worst false positives and 100 worst false negatives as readable side-by-side text, with the feature values that drove each decision. Read them yourself, then post the patterns you find — that is how the team learns what feature is missing, and it fills section 5 of the required methodology document.

### Kickoff prompt for your agent

```markdown
I am building the evaluation and feature layer for an entity-resolution
task scored by macro-averaged F-beta with beta=0.5.

1. metric.py FIRST. Per entity, F = 1.25*c / (0.25*k + m) where c is the
   count of correct predicted IDs, k the count of true IDs, m the count
   predicted. An entity with k=0 scores 1.0 if m=0 and 0.0 otherwise.
   Singletons are included in the macro average. Missing entities raise.
   Unit test: predicting [A,B,C] against truth [A,C] must return 0.714.

2. features.py. featurise(s1_rows, cand_rows, context) -> float32 array,
   columns in the exact order of FEATURE_NAMES, where each entry carries a
   name and a monotonic direction (+1, -1 or 0). Groups: name similarity,
   address similarity, exact-match indicators, context (blocking channel
   count, rank within entity, margin to best), and competition features
   (a candidate's best score across all entities that proposed it).
   Use rapidfuzz, vectorised where possible. Must handle empty addresses
   and non-Latin scripts without crashing.

3. s7_evaluate.py. JSON report: overall F0.5, precision and recall
   separately, broken down by country and by true-match-count bucket,
   singleton accuracy, and blocking recall ceiling.

4. error_dump.py. The 100 worst false positives and false negatives as
   readable side-by-side text with their feature values.

Tests before implementation on metric.py. It is the number the whole team
trusts, so it has to be provably right.
```

## Timeline

Roughly 52 hours remain. Six gates, each ending in something that works. The governing rule: **a valid submission must exist by Gate 1, on Friday night.** A team with a mediocre submission at hour 9 and a great one at hour 48 finishes well. A team with a brilliant pipeline that first produces output at hour 50 finishes nowhere.

| Gate | By when | What must exist |
| --- | --- | --- |
| G0 | Fri 22:30 | Repo, schemas, stubs, smoke sample, validation split, the country answer |
| G1 | Sat 04:00 | End-to-end run producing a validated submission. First upload. |
| G2 | Sat 16:00 | Full blocking with measured recall, LightGBM v1, calibration, transfer test |
| G3 | Sun 04:00 | Embedding channel, competition features, expected-F₀.₅ selection live |
| G4 | Sun 14:00 | Feature freeze. Final training run, full test inference. |
| G5 | Sun 20:00 | Zip packaged, methodology written, validator passing. Four-hour buffer. |

### Who is doing what, per cycle

|  | Nidhi | Tanuj | Parth | Krrish |
| --- | --- | --- | --- | --- |
| **To G0** | Repo, ingest, split, stubs, country check | Environment, Kaggle dataset upload | Acceptance-test file from the noise catalogue | `metric.py` + its unit test |
| **To G1** | Exact-key and rare-token channels; naive assembly | Trivial model on 10 features to close the loop | Normalisation v1 on the smoke sample | `featurise` v1, 20 features |
| **To G2** | TF-IDF channels at full scale; recall report | Training set, LightGBM v1, calibration, transfer test | Full-scale normalisation, lexicons, mined table | Full feature set, evaluation report |
| **To G3** | Uniqueness constraint, expected-F₀.₅ search | Embeddings, ANN index, cosine feature | Mined lexicon v2 from error analysis | Competition features, error dumps |
| **To G4** | Final inference, threshold tuning, uploads | Final training, ensemble if time | Methodology sections 2 and 3 | Methodology sections 5 and 6 |
| **To G5** | Package the zip, run the validator | `requirements.txt`, reproduction README | Lexicon documentation | Final report, charts |

### Two rules that keep four people from colliding

**Nobody waits on a real implementation.** Stubs exist from Gate 0, so every track always has something to run against. If Parth's normalisation is not ready, the stub passes strings through unchanged and Krrish's features still compute. A blocked person is a wasted person.

**Merge to `main` only through a green smoke run.** Two minutes on the 50k sample, every time, no exceptions. On a 52-hour clock you cannot afford to spend three hours discovering that a merge broke the pipeline at 3 a.m.

### Coverage

Stagger rest rather than all four working straight through. Two of you offline roughly 02:00–07:00 and the other two roughly 06:00–11:00 means the pipeline always has an owner awake, and decisions do not queue up waiting for someone to wake. Tired people merge broken code, and this plan has no slack for a bad merge on Sunday morning.

## Submission strategy

Five uploads per day across three days, so fifteen in total, and the button disables afterwards. Two of today's are worth spending on information rather than score.

### The free measurement nobody takes

**Submit an all-empty prediction** — every test entity present, every `matched_entity_ids` blank. It is a perfectly valid submission, and because an entity with no true matches scores 1.0 for an empty list while every other entity scores 0.0, **the returned score is exactly the singleton rate of the public test subset**.

That single number is worth having. It confirms the format is accepted before you have anything real to lose, and it gives the selection layer a measured prior for P(k = 0) on the actual test distribution instead of the 5.58% inherited from training — which may well differ, since France is 15% of test and behaves unlike anything in the training data.

A second cheap probe, if you can spare it at Gate 1: **top-1 candidate for every entity**. Compare the result against the same prediction scored on your validation split. A large gap means your validation is optimistic, and better to learn that on Friday than on Sunday.

### Rules for the other thirteen

- **Run the organisers' validator before every upload, without exception.** A rejected file costs a full attempt and returns nothing.
- **Never upload something you have not scored locally first.** Validation on 400k held-out entities is a much larger sample than the public leaderboard subset, so it is the more trustworthy number.
- **Do not chase the public leaderboard.** Final ranking comes from the private split. If local validation and the public board disagree, follow validation — unless the gap is enormous, which means a bug, not overfitting.
- **Pace yourself across Sunday.** Keep at least two of Sunday's five for after the final training run. Teams that burn all five by noon have nothing left when the last idea works.
- **Change one thing at a time** once you are past Gate 2. An upload that moves the score without a known cause teaches you nothing.

### The submission log

The guidelines require version history — shortlisting is based on submitted solutions, and the top 100 teams are asked for their source. Keep `submissions.md` in the repo, updated at upload time, not afterwards:

| # | Time | Git SHA | Change | Local F₀.₅ | Public LB |
| --- | --- | --- | --- | --- | --- |
| 1 | Fri 23:40 | `a1b2c3d` | all-empty probe | — |  |
| 2 | Sat 04:10 | `e4f5g6h` | end-to-end baseline |  |  |

The gap between the local and leaderboard columns is the most useful diagnostic you will have all weekend. Watch it widen or narrow as you go.

## Risks

The point of naming these now is that the fallback gets decided while everyone is still rested.

| Risk | How you notice | What to do |
| --- | --- | --- |
| Blocking recall far below 97% | Nidhi's G2 recall report | Add channels and raise the cap before touching the model. No classifier beats the ceiling. |
| Feature matrix will not fit on disk | A 25 GB write fails mid-run | Fuse featurise and score, shard at 5M pairs, keep only probabilities. Build it this way from the start. |
| Kaggle session dies mid-embedding | Notebook stops with no output | Checkpoint every 500k records and resume from the last shard. |
| France collapses the score | Tanuj's US-to-India transfer test at G2 | Drop country features, tighten monotonic constraints, lean on character-level and embedding similarity. |
| Indic text arrives as question marks | Parth's acceptance test fails on cross-script pairs | Set `PYTHONUTF8=1` and pass `encoding="utf-8"` everywhere. Verify at Gate 0. |
| Final inference outruns the clock | Runtime still unknown at Gate 4 | Time the full test inference at Gate 3, while shrinking the candidate cap is still an option. |
| A track stalls | No merge from that branch in six hours | The Gate 0 stub keeps the pipeline running. Redistribute the missing piece, not the whole track. |
| Submission rejected on format | Validator output | Run `validate_submission.py` before every upload. |
| Local score and leaderboard diverge | The gap column in `submissions.md` | Suspect a bug before suspecting overfitting. Check ID coverage and singleton handling first. |
| Disk fills mid-run | Write errors on `artifacts/` | Delete superseded stage outputs at each gate. Keep only the newest of each stage. |

The two that actually end hackathons are the last-minute integration failure and the final run that does not finish in time. Both are prevented by the same discipline: frozen contracts from Gate 0, and a full-scale timing test at Gate 3 rather than an optimistic estimate at Gate 4.

## Final package

Every team submits one zip alongside the leaderboard uploads, and **the top teams' packages are reviewed in detail before final rankings are confirmed**. A strong score with a weak package is how a team drops out of the top 100 at the last step.

```markdown
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv      # the file you uploaded to the portal
│   └── candidate_pairs.tsv       # the blocking set fed to the model
├── code/business_entity_resolution/
│   ├── src/
│   ├── README.md                 # exact end-to-end reproduction steps
│   └── requirements.txt          # pinned versions
└── Documentation_template.md     # filled in, keep the filename
```

### The methodology document

The template ships in `student_resource/`. There is no page limit and the brief asks for technical depth over brevity, so write it properly — it is the artefact that distinguishes a team that understood the problem from one that ran a library. Four things belong in it that most teams will not have:

1. **The metric derivation.** Show that F₀.₅ reduces to `1.25c / (0.25k + m)`, and that the optimal threshold is therefore `c / (0.25k + m)` and rises with list length. Then show the expected-F₀.₅ prefix search that follows from it. This demonstrates you optimised the actual objective rather than a proxy.
2. **The many-to-one discovery.** State that all 7,638,365 matched IDs in the ground truth are distinct, that this implies each Source-2/3 record belongs to at most one Source-1 entity, and show how the uniqueness constraint and the competition features exploit it. This is a genuine finding from the data, and reviewers notice findings.
3. **The mined transliteration lexicon.** Explain the PMI alignment over ground-truth groups, show sample mappings, and quantify the recall it recovers on Indian pairs against generic romanisation alone. Say plainly that it derives only from the provided training data.
4. **The blocking recall analysis.** Recall ceiling, reduction ratio, and the drop-one contribution of each channel. Section 3 of the template asks exactly this.

Also report the honest numbers: the US-to-India transfer result as a France estimate, and what your error analysis found. A write-up that names its own weaknesses reads as competent, not weak.

### The README must actually reproduce

Someone else has to regenerate both output files from the raw data using only what is in that folder. Write the literal commands in order, state the expected runtime per stage and the hardware you ran on, and pin every version. Then have someone who did not write the pipeline follow it from a clean checkout. If that takes more than ten minutes to discover a problem, it was worth doing.

### Final checklist, Sunday evening

- [ ] `validate_submission.py` prints `PASS` on the exact files in the zip
- [ ] `matching_results.tsv` has 1,732,545 lines including the header
- [ ] Every ID in the results also appears in that entity's `candidate_pairs.tsv` row
- [ ] No `S1-` IDs appear in any match list, and no IDs absent from the test files
- [ ] Singleton rows are genuinely empty, not `nan`
- [ ] Both files are tab-separated and UTF-8
- [ ] `requirements.txt` pins every version, and the model's licence is named in the write-up
- [ ] `submissions.md` version history is complete
- [ ] The zip opens to exactly the structure above, with `Documentation_template.md` still named that

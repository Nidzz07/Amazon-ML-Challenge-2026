"""S4 Train (owner: Tanuj): features_train -> model.txt, calibrator.pkl.

Keeps all positives, samples hard negatives from top-ranked non-matches at 2:1,
subsamples to 40% of entities, splits by ENTITY (not pair), trains LightGBM with
monotonic constraints, then fits isotonic calibration on a disjoint third slice.

Usage:
    python s4_train.py [--smoke] [--input DIR] [--output DIR]
    python s4_train.py [--smoke] [--input DIR] [--output DIR] --neg-ratio 2.0 --entity-frac 0.4
"""
import json
import pickle
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

import config
import pipeline_io as pio

# ── Sampling knobs (Gate 2: tune after first end-to-end run) ─────────────────
NEG_TO_POS_RATIO = 2.0          # hard negatives per positive
ENTITY_SUBSAMPLE_FRAC = 0.40    # fraction of entities to keep (memory budget)
VAL_FRAC = 0.20                 # entity fraction for LGB early-stopping
CALIB_FRAC = 0.10               # entity fraction for isotonic calibration (disjoint)

# ── LightGBM base params (tune only after the pipeline is closed end-to-end) ─
LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "seed": config.SEED,
    "verbosity": -1,
}
LGB_ROUNDS = 500
LGB_EARLY_STOP = 50


# ─────────────────────────────────────────────────────────────────────────────
def _label_candidates(lf: pl.LazyFrame, gt: pl.LazyFrame) -> pl.LazyFrame:
    """Join blocking candidates with ground truth to create the label column."""
    positives = (
        gt
        .select("source1_entity_id", "candidate_entity_id")
        .with_columns(pl.lit(1, dtype=pl.UInt8).alias("label"))
    )
    return lf.join(positives, on=["source1_entity_id", "candidate_entity_id"], how="left") \
             .with_columns(pl.col("label").fill_null(pl.lit(0, dtype=pl.UInt8)))


def _sample_entities(entities: list, frac: float, seed: int) -> set:
    rng = np.random.default_rng(seed)
    k = max(1, int(len(entities) * frac))
    return set(rng.choice(entities, size=k, replace=False).tolist())


def _entity_split(df: pl.DataFrame, val_frac: float, calib_frac: float, seed: int):
    """Return (train_df, val_df, calib_df) split strictly by entity."""
    entities = df["source1_entity_id"].unique().to_list()
    rng = np.random.default_rng(seed)
    rng.shuffle(entities)

    n = len(entities)
    n_calib = max(1, int(n * calib_frac))
    n_val   = max(1, int(n * val_frac))

    calib_ids = set(entities[:n_calib])
    val_ids   = set(entities[n_calib:n_calib + n_val])

    calib = df.filter(pl.col("source1_entity_id").is_in(calib_ids))
    val   = df.filter(pl.col("source1_entity_id").is_in(val_ids))
    train = df.filter(~pl.col("source1_entity_id").is_in(calib_ids | val_ids))
    return train, val, calib


def _sample_negatives(df: pl.DataFrame, n_pos: int, ratio: float, seed: int, feature_names: tuple[str, ...]) -> pl.DataFrame:
    """Keep all positives; sample hard negatives (top prior_score) at ratio:1."""
    if "prior_score" in feature_names:
        score_col = f"f{feature_names.index('prior_score'):03d}"
    else:
        score_col = "f000"
    neg = df.filter(pl.col("label") == 0).sort(score_col, descending=True)
    target = int(n_pos * ratio)
    if len(neg) > target:
        neg = neg.head(target)
    return neg


def main(argv=None) -> None:
    ap = pio.parser(__doc__)
    ap.add_argument("--neg-ratio", type=float, default=NEG_TO_POS_RATIO)
    ap.add_argument("--entity-frac", type=float, default=ENTITY_SUBSAMPLE_FRAC)
    ap.add_argument("--transfer-test", action="store_true", help="Run the France proxy transfer test and exit")
    args = ap.parse_args(argv)
    in_dir, out_dir = pio.dirs(args)
    t0 = time.perf_counter()

    feature_names, feature_version = pio.feature_spec()
    feature_cols = pio.feature_columns(len(feature_names))
    print(f"Feature version: {feature_version}  |  {len(feature_cols)} features")

    # ── 1. Load features + labels ─────────────────────────────────────────
    print("Loading features_train ...")
    feat_lf = pl.scan_parquet(config.features_path("train", in_dir))

    # Exclude held-out validation entities (Nidhi's validation split)
    held_out = pio.val_ids(args.smoke)
    if held_out is not None:
        feat_lf = feat_lf.filter(~pl.col("source1_entity_id").is_in(held_out))

    df = feat_lf.collect()
    print(f"  Loaded {len(df):,} rows, {df['label'].mean():.4f} positive rate")

    if args.transfer_test:
        print("\n--- Running France Proxy Transfer Test ---")
        s1_meta = pl.read_parquet(config.records_path("train", "config.SOURCE1_SRC" if not hasattr(config, "SOURCE1_SRC") else config.SOURCE1_SRC, in_dir), columns=["entity_id", "country"]) if hasattr(config, "SOURCE1_SRC") else pl.read_parquet(config.records_path("train", "source1", in_dir), columns=["entity_id", "country"])
        df = df.join(s1_meta.rename({"entity_id": "source1_entity_id"}), on="source1_entity_id", how="left")
        
        us_df = df.filter(pl.col("country") == "US")
        in_df = df.filter(pl.col("country") == "India")
        
        us_pos = us_df.filter(pl.col("label") == 1)
        us_neg = _sample_negatives(us_df, len(us_pos), args.neg_ratio, config.SEED, feature_names)
        us_df = pl.concat([us_pos, us_neg])
        
        in_pos = in_df.filter(pl.col("label") == 1)
        in_neg = _sample_negatives(in_df, len(in_pos), args.neg_ratio, config.SEED, feature_names)
        in_df = pl.concat([in_pos, in_neg])
        
        try:
            from features import FEATURE_MONO
            mono = list(FEATURE_MONO)
        except (ImportError, AttributeError):
            mono = [0] * len(feature_cols)
        params = {**LGB_PARAMS, "monotone_constraints": mono}
        
        def train_and_eval(tr_df, te_df, tr_name, te_name):
            X_tr, y_tr = tr_df.select(feature_cols).to_numpy(), tr_df["label"].to_numpy()
            X_te, y_te = te_df.select(feature_cols).to_numpy(), te_df["label"].to_numpy()
            dtr = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols, free_raw_data=False)
            model = lgb.train(params, dtr, num_boost_round=100)
            preds = model.predict(X_te)
            auc = roc_auc_score(y_te, preds)
            print(f"  Train: {tr_name:5s} | Eval: {te_name:13s} | AUC: {auc:.4f}")
            return auc
            
        print("  Evaluating US -> India (transfer proxy)")
        train_and_eval(us_df, us_df, "US", "US (in-dist)")
        train_and_eval(us_df, in_df, "US", "India")
        
        print("\n  Evaluating India -> US (transfer proxy)")
        train_and_eval(in_df, in_df, "India", "India (in-dist)")
        train_and_eval(in_df, us_df, "India", "US")
        return

    # ── 2. Entity subsample ───────────────────────────────────────────────
    all_entities = df["source1_entity_id"].unique().to_list()
    sampled = _sample_entities(all_entities, args.entity_frac, config.SEED)
    df = df.filter(pl.col("source1_entity_id").is_in(sampled))
    print(f"  After entity subsample ({args.entity_frac:.0%}): {len(df):,} rows, "
          f"{df['source1_entity_id'].n_unique():,} entities")

    # ── 3. Hard-negative sampling ─────────────────────────────────────────
    pos = df.filter(pl.col("label") == 1)
    neg = _sample_negatives(df, len(pos), args.neg_ratio, config.SEED, feature_names)
    df = pl.concat([pos, neg])
    print(f"  After neg sampling ({args.neg_ratio:.1f}:1): {len(df):,} rows  "
          f"({len(pos):,} pos / {len(neg):,} neg)")

    # ── 4. Entity-level train / val / calib split ─────────────────────────
    train_df, val_df, calib_df = _entity_split(df, VAL_FRAC, CALIB_FRAC, config.SEED)
    print(f"  Train: {len(train_df):,}  Val: {len(val_df):,}  Calib: {len(calib_df):,}")

    # ── 5. Build LightGBM datasets ────────────────────────────────────────
    # Monotonic constraints: +1 for similarity/channel features, 0 for unconstrained.
    try:
        from features import FEATURE_MONO
        mono = list(FEATURE_MONO)
    except (ImportError, AttributeError):
        mono = [0] * len(feature_cols)

    params = {**LGB_PARAMS, "monotone_constraints": mono}

    X_tr, y_tr = train_df.select(feature_cols).to_numpy(), train_df["label"].to_numpy()
    X_va, y_va = val_df.select(feature_cols).to_numpy(),   val_df["label"].to_numpy()

    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols, free_raw_data=False)
    dval   = lgb.Dataset(X_va, label=y_va, feature_name=feature_cols, reference=dtrain)

    # ── 6. Train ──────────────────────────────────────────────────────────
    print("Training LightGBM ...")
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=LGB_ROUNDS,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(LGB_EARLY_STOP), lgb.log_evaluation(period=25)],
    )

    val_preds = model.predict(X_va)
    print(f"  Val AUC  : {roc_auc_score(y_va, val_preds):.4f}")
    print(f"  Val Brier: {brier_score_loss(y_va, val_preds):.5f}")

    # ── 7. Isotonic calibration on the disjoint calib slice ───────────────
    print("Fitting isotonic calibration ...")
    X_ca, y_ca = calib_df.select(feature_cols).to_numpy(), calib_df["label"].to_numpy()
    raw_calib_preds = model.predict(X_ca)
    ir = IsotonicRegression(out_of_bounds="clip")
    cal_preds = ir.fit_transform(raw_calib_preds, y_ca)

    print(f"  Calib Brier (raw) : {brier_score_loss(y_ca, raw_calib_preds):.5f}")
    print(f"  Calib Brier (cal) : {brier_score_loss(y_ca, cal_preds):.5f}")

    # ── 8. Save model.txt (as LightGBM text + metadata header) ───────────
    model_out = config.model_path(out_dir)
    model.save_model(str(model_out))

    # Embed metadata in a sidecar so s5_score can guard feature version
    meta = {
        "kind": "lightgbm",
        "feature_version": feature_version,
        "feature_names": list(feature_names),
        "best_iteration": model.best_iteration,
        "train_rows": len(train_df),
        "pos_rate": float(train_df["label"].mean()),
        "val_auc": float(roc_auc_score(y_va, val_preds)),
        "seed": config.SEED,
    }
    model_out.with_suffix(".meta").write_text(json.dumps(meta, indent=2), encoding=config.ENCODING)

    # ── 9. Save calibrator ────────────────────────────────────────────────
    with open(config.calibrator_path(out_dir), "wb") as f:
        pickle.dump(ir, f)

    print(f"model.txt        -> {model_out}")
    print(f"model.meta       -> {model_out.with_suffix('.meta')}")
    print(f"calibrator.pkl   -> {config.calibrator_path(out_dir)}")
    print(f"s4 done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())

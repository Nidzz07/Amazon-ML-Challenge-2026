"""S4 Train (owner: Tanuj): features_train -> model.txt, model.meta, calibrator.pkl.

Streams features_train by entity, never loading the whole matrix (66M rows x 71 features is ~19 GB).
Every train entity is used (no entity subsample). Entities are split by a hash of their id:

    train (70%)  all positives + the top-N hard negatives PER ENTITY by prior_score  -> LightGBM fit
    val   (20%)  same sampling as train                                              -> early stopping
    calib (10%)  UNSAMPLED: every capped candidate of the entity, as at inference    -> isotonic fit

The isotonic calibrator is fitted on the unsampled slice, so it learns the real inference prevalence
(~7.7 negatives per positive) even though the model saw ~2:1. Held-out validation entities
(pio.val_ids) are excluded everywhere and never touched.

Usage:
    python s4_train.py [--smoke] [--input DIR] [--output DIR]
    python s4_train.py --neg-ratio 2 --min-negs 3 --val-frac 0.2 --calib-frac 0.1 --rounds 800
    python s4_train.py --transfer-test        # France proxy: train US -> eval India and back
"""
import json
import pickle
import sys
import time
from typing import Iterator

import lightgbm as lgb
import numpy as np
import polars as pl
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

import config
import pipeline_io as pio

NEG_TO_POS_RATIO = 2.0      # hard negatives kept per positive, per entity
MIN_NEGS = 3                # floor per entity, so entities with no true match still contribute negatives
VAL_FRAC = 0.20             # entity share used for early stopping
CALIB_FRAC = 0.10           # entity share kept unsampled for the isotonic fit
BATCH_ROWS = 1_000_000      # rows per streamed batch
LGB_ROUNDS = 500
LGB_EARLY_STOP = 50

LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "seed": config.SEED,
    "verbosity": -1,
}

ID = "source1_entity_id"


def rss_gb(tag: str) -> None:
    """Print current and peak process memory (peak is Windows-only via psutil, else current)."""
    try:
        import psutil
        i = psutil.Process().memory_info()
        print(f"    [mem] {tag}: rss {i.rss / 2**30:.2f} GB  peak {getattr(i, 'peak_wset', i.rss) / 2**30:.2f} GB", flush=True)
    except ImportError:
        pass
TRAIN, VAL, CALIB = 0, 1, 2


# ── streaming and sampling primitives ────────────────────────────────────────
def _row_group_batches(path, columns: list[str], batch_rows: int) -> Iterator[pl.DataFrame]:
    """Read row group by row group and emit frames of >= batch_rows rows. Unlike ParquetFile.iter_batches this
    keeps memory bounded by one batch however large the file is (iter_batches held >1 GB of Arrow buffers)."""
    pf = pq.ParquetFile(str(path))
    pending, rows = [], 0
    for g in range(pf.num_row_groups):
        pending.append(pl.from_arrow(pf.read_row_group(g, columns=columns)))
        rows += pending[-1].height
        if rows >= batch_rows:
            yield pl.concat(pending, rechunk=True)
            pending, rows = [], 0
    if pending:
        yield pl.concat(pending, rechunk=True)


def iter_entity_batches(path, columns: list[str], batch_rows: int = BATCH_ROWS) -> Iterator[pl.DataFrame]:
    """Yield frames that never cut an entity in half (a trailing run of one entity is carried to the next
    batch). Assumes an entity's rows are contiguous, which s3's per-entity blocks give; if they were not,
    the only effect is that a split entity is treated as two."""
    carry = None
    for df in _row_group_batches(path, columns, batch_rows):
        if carry is not None:
            df = pl.concat([carry, df], rechunk=True)
        ids = df[ID]
        differs = np.flatnonzero((ids != ids[-1]).to_numpy())
        cut = int(differs[-1]) + 1 if len(differs) else 0
        if cut:
            yield df[:cut]
        carry = df[cut:]
    if carry is not None and carry.height:
        yield carry


def entity_bucket(ids: pl.Series, seed: int = config.SEED) -> np.ndarray:
    """Stable 0..99 bucket per entity id, independent of batch boundaries."""
    return (ids.hash(seed=seed) % 100).to_numpy().astype(np.uint8)


def assign_role(bucket: np.ndarray, val_frac: float, calib_frac: float) -> np.ndarray:
    n_calib, n_val = round(calib_frac * 100), round(val_frac * 100)
    role = np.full(bucket.shape, TRAIN, dtype=np.uint8)
    role[bucket < n_calib + n_val] = VAL
    role[bucket < n_calib] = CALIB
    return role


def select_hard_negatives(df: pl.DataFrame, prior_col: str, ratio: float, min_negs: int) -> pl.DataFrame:
    """Keep every positive, plus per ENTITY the top negatives by prior_score: ceil(ratio * n_pos), at least
    min_negs. Per-entity is what inference sees (each entity competes only against its own candidates)."""
    ranked = df.with_columns(
        pl.col("label").sum().over(ID).alias("_npos"),
        pl.when(pl.col("label") == 0).then(pl.col(prior_col)).rank("ordinal", descending=True).over(ID).alias("_nr"),
    )
    quota = (pl.col("_npos") * ratio).ceil().clip(lower_bound=min_negs)
    return ranked.filter((pl.col("label") == 1) | (pl.col("_nr") <= quota)).drop("_npos", "_nr")


def _stack(parts: list[np.ndarray], dtype) -> np.ndarray:
    """Concatenate into ONE preallocated array, freeing each part as it is copied, so peak memory is ~1x the data
    (np.concatenate keeps the parts and the result alive together, ~2x)."""
    if not parts:
        return np.empty((0,), dtype=dtype)
    out = np.empty((sum(len(p) for p in parts), *parts[0].shape[1:]), dtype=dtype)
    i = 0
    while parts:
        p = parts.pop(0)
        out[i:i + len(p)] = p
        i += len(p)
    return out


def load_slices(path, feature_cols: list[str], prior_col: str, held_out, ratio: float, min_negs: int,
                val_frac: float, calib_frac: float, batch_rows: int = BATCH_ROWS) -> dict:
    """One streamed pass -> {role: dict(X float32, y uint8, bucket uint8)}. Peak memory is the kept rows only."""
    cols = [ID, "label", *feature_cols]
    acc = {r: {"X": [], "y": [], "b": []} for r in (TRAIN, VAL, CALIB)}
    seen = kept = 0
    t0 = time.perf_counter()
    for i, df in enumerate(iter_entity_batches(path, cols, batch_rows)):
        if held_out is not None:
            df = df.filter(~pl.col(ID).is_in(held_out.implode()))
        seen += df.height
        if not df.height:
            continue
        bucket = entity_bucket(df[ID])
        role = assign_role(bucket, val_frac, calib_frac)
        for r in (TRAIN, VAL, CALIB):
            m = role == r
            if not m.any():
                continue
            part = df.filter(pl.Series(m))
            b = bucket[m]
            if r != CALIB:
                part = select_hard_negatives(part, prior_col, ratio, min_negs)
                b = entity_bucket(part[ID])
            acc[r]["X"].append(part.select(feature_cols).to_numpy(order="c").astype(np.float32, copy=False))
            acc[r]["y"].append(part["label"].to_numpy().astype(np.uint8))
            acc[r]["b"].append(b)
            kept += part.height
        if i % 10 == 0:
            print(f"    batch {i:4d}: read {seen:>12,} rows  kept {kept:>11,}  ({time.perf_counter() - t0:.0f}s)", flush=True)
    out = {r: {"X": _stack(a["X"], np.float32), "y": _stack(a["y"], np.uint8), "b": _stack(a["b"], np.uint8)}
           for r, a in acc.items()}
    print(f"  streamed {seen:,} rows in {time.perf_counter() - t0:.0f}s")
    return out


# ── calibration ──────────────────────────────────────────────────────────────
def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error over equal-count bins."""
    order = np.argsort(p)
    total = 0.0
    for idx in np.array_split(order, bins):
        if len(idx):
            total += len(idx) * abs(float(p[idx].mean()) - float(y[idx].mean()))
    return total / len(y)


def reliability_table(y: np.ndarray, p: np.ndarray, edges=None) -> list[dict]:
    """Per probability bin: rows, mean predicted, observed match rate. Empty bins are omitted."""
    edges = np.linspace(0.0, 1.0, 11) if edges is None else np.asarray(edges)
    which = np.digitize(p, edges) - 1                    # bins are [lo, hi); the top edge itself joins the last bin
    which[p == edges[-1]] = len(edges) - 2
    out = []
    for b in range(len(edges) - 1):
        m = which == b                                   # values outside [edges[0], edges[-1]] fall in no bin
        if m.any():
            out.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": int(m.sum()),
                        "mean_pred": float(p[m].mean()), "observed": float(y[m].mean())})
    return out


def fit_isotonic_with_check(raw: np.ndarray, y: np.ndarray, bucket: np.ndarray):
    """Final isotonic fit on all calib rows, plus an honest 2-fold entity-level check (fit on one half of the
    calib entities, score the other) because fitting and scoring on the same rows flatters isotonic."""
    fold = (bucket % 2).astype(bool)
    cv = []
    p_cv = np.full(len(raw), np.nan)                       # every row scored by a calibrator that never saw its entity
    for train_mask in (fold, ~fold):
        if train_mask.sum() == 0 or (~train_mask).sum() == 0:
            continue
        ir = IsotonicRegression(out_of_bounds="clip").fit(raw[train_mask], y[train_mask])
        p = ir.transform(raw[~train_mask])
        p_cv[~train_mask] = p
        cv.append((brier_score_loss(y[~train_mask], p), ece(y[~train_mask], p)))
    final = IsotonicRegression(out_of_bounds="clip").fit(raw, y)
    stats = {
        "prevalence": float(y.mean()),
        "neg_per_pos": float((1 - y.mean()) / max(y.mean(), 1e-12)),
        "mean_raw": float(raw.mean()),
        "brier_raw": float(brier_score_loss(y, raw)),
        "ece_raw": float(ece(y, raw)),
        "brier_cv": float(np.mean([c[0] for c in cv])) if cv else None,
        "ece_cv": float(np.mean([c[1] for c in cv])) if cv else None,
    }
    ok = ~np.isnan(p_cv)
    if ok.any():
        # cross-fitted, so this is an honest read of "when we say 0.7, how often is it a match?"
        stats["reliability"] = reliability_table(y[ok], p_cv[ok])
        stats["near_0_7"] = next(iter(reliability_table(y[ok], p_cv[ok], [0.65, 0.75])), None)
    return final, stats


def _predict(model, X: np.ndarray, num_iteration=None, chunk: int = 2_000_000) -> np.ndarray:
    return np.concatenate([model.predict(X[i:i + chunk], num_iteration=num_iteration) for i in range(0, len(X), chunk)])


def monotone_constraints(n_features: int) -> list[int]:
    try:
        from features import FEATURE_MONO
        mono = list(FEATURE_MONO)
    except (ImportError, AttributeError):
        mono = [0] * n_features
    assert len(mono) == n_features, f"{len(mono)} monotone constraints for {n_features} features"
    return mono


def print_reliability(cal: dict) -> None:
    """Cross-fitted reliability table; the 0.65-0.75 bin is the one the selection layer leans on."""
    if not cal.get("reliability"):
        return
    print("  reliability (cross-fitted calibrated prob vs observed match rate):")
    print(f"    {'bin':>11s} {'rows':>10s} {'mean pred':>10s} {'observed':>9s}")
    for r in cal["reliability"]:
        print(f"    {r['lo']:4.2f}-{r['hi']:4.2f} {r['n']:>10,} {r['mean_pred']:>10.3f} {r['observed']:>9.3f}")
    b = cal.get("near_0_7")
    if b:
        print(f"  near 0.7 (0.65-0.75): {b['n']:,} rows, mean pred {b['mean_pred']:.3f}, observed {b['observed']:.3f}")
    else:
        print("  near 0.7 (0.65-0.75): no rows fall in this bin")


# ── France proxy: train on one country, evaluate on the other ────────────────
def transfer_test(path, in_dir, feature_cols, prior_col, held_out, args, mono) -> None:
    s1 = pl.read_parquet(config.records_path("train", config.SOURCE1_SRC, in_dir), columns=["entity_id", "country"])
    s1 = s1.rename({"entity_id": ID})
    countries = ("US", "India")
    sets = {c: {"X": [], "y": [], "b": []} for c in countries}
    for df in iter_entity_batches(path, [ID, "label", *feature_cols], args.batch_rows):
        if held_out is not None:
            df = df.filter(~pl.col(ID).is_in(held_out.implode()))
        if args.transfer_frac < 1.0:      # keep RAM bounded at full scale: a stable slice of entities is plenty for a proxy
            df = df.filter(pl.Series(entity_bucket(df[ID]) < round(args.transfer_frac * 100)))
        df = df.join(s1, on=ID, how="left")
        for c in countries:
            part = df.filter(pl.col("country") == c)
            if part.height:
                part = select_hard_negatives(part, prior_col, args.neg_ratio, args.min_negs)
                sets[c]["X"].append(part.select(feature_cols).to_numpy(order="c").astype(np.float32, copy=False))
                sets[c]["y"].append(part["label"].to_numpy())
                sets[c]["b"].append(entity_bucket(part[ID]))
    data = {c: tuple(np.concatenate(s[k]) for k in ("X", "y", "b")) for c, s in sets.items()}
    params = {**LGB_PARAMS, "monotone_constraints": mono}

    def fit(X, y):
        return lgb.train(params, lgb.Dataset(X, label=y, feature_name=feature_cols), num_boost_round=args.transfer_rounds)

    print(f"\n  {'train':6s} {'eval':12s} {'AUC':>7s} {'Brier':>8s}")
    for src, dst in (("US", "India"), ("India", "US")):
        Xs, ys, bs = data[src]
        # in-distribution eval on ENTITIES the model never saw: entity-hash holdout inside the source country
        hold = bs % 5 == 0
        m = fit(Xs[~hold], ys[~hold])
        for name, (X, y) in ((f"{src} (held)", (Xs[hold], ys[hold])), (dst, data[dst][:2])):
            p = m.predict(X)
            print(f"  {src:6s} {name:12s} {roc_auc_score(y, p):7.4f} {brier_score_loss(y, p):8.5f}")
    if not args.no_country_variant:
        country_variant(data, params, feature_cols, args.transfer_rounds)


def country_variant(data: dict, params: dict, feature_cols: list[str], rounds: int) -> None:
    """Does a country feature help at all? Train one US+India model without it and one with it, and score both on
    held-out ENTITIES of each country. France has no training data, so a country feature can never be learned for it;
    if it buys little in-distribution, the country-blind model is the safer choice."""
    (Xu, yu, bu), (Xi, yi, bi) = data["US"], data["India"]
    X, y, b = np.concatenate([Xu, Xi]), np.concatenate([yu, yi]), np.concatenate([bu, bi])
    is_india = np.concatenate([np.zeros(len(yu), dtype=bool), np.ones(len(yi), dtype=bool)])
    hold = b % 5 == 0
    print(f"\n  Country feature (US+India model, {int((~hold).sum()):,} train rows; eval on held-out entities)")
    print(f"  {'model':14s} {'eval':6s} {'AUC':>7s} {'Brier':>8s}")
    results = {}
    for name, mat, names, mono in (("country-blind", X, feature_cols, params["monotone_constraints"]),
                                   ("country-aware", None, [*feature_cols, "country_is_india"],
                                    [*params["monotone_constraints"], 0])):
        if mat is None:
            mat = np.column_stack([X, is_india.astype(np.float32)]).astype(np.float32)
        m = lgb.train({**params, "monotone_constraints": mono},
                      lgb.Dataset(mat[~hold], label=y[~hold], feature_name=names), num_boost_round=rounds)
        for cname, mask in (("US", ~is_india), ("India", is_india)):
            sel = hold & mask
            p = m.predict(mat[sel])
            results[(name, cname)] = (roc_auc_score(y[sel], p), brier_score_loss(y[sel], p))
            print(f"  {name:14s} {cname:6s} {results[(name, cname)][0]:7.4f} {results[(name, cname)][1]:8.5f}")
        del mat
    for cname in ("US", "India"):
        da = results[("country-aware", cname)][0] - results[("country-blind", cname)][0]
        print(f"  AUC gain from the country feature on {cname}: {da:+.4f}")


# ─────────────────────────────────────────────────────────────────────────────
def main(argv=None) -> None:
    ap = pio.parser(__doc__)
    ap.add_argument("--neg-ratio", type=float, default=NEG_TO_POS_RATIO, help="hard negatives per positive, per entity")
    ap.add_argument("--min-negs", type=int, default=MIN_NEGS, help="minimum negatives kept per entity")
    ap.add_argument("--val-frac", type=float, default=VAL_FRAC)
    ap.add_argument("--calib-frac", type=float, default=CALIB_FRAC)
    ap.add_argument("--batch-rows", type=int, default=BATCH_ROWS)
    ap.add_argument("--rounds", type=int, default=LGB_ROUNDS)
    ap.add_argument("--early-stop", type=int, default=LGB_EARLY_STOP)
    ap.add_argument("--transfer-test", action="store_true", help="France proxy: US -> India and India -> US, then exit")
    ap.add_argument("--transfer-rounds", type=int, default=100)
    ap.add_argument("--transfer-frac", type=float, default=0.25, help="share of entities (by hash) used by --transfer-test")
    ap.add_argument("--no-country-variant", action="store_true", help="skip the country-blind vs country-aware comparison")
    args = ap.parse_args(argv)
    in_dir, out_dir = pio.dirs(args)
    t0 = time.perf_counter()

    feature_names, feature_version = pio.feature_spec()
    feature_cols = pio.feature_columns(len(feature_names))
    prior_col = feature_cols[feature_names.index("prior_score")] if "prior_score" in feature_names else feature_cols[0]
    mono = monotone_constraints(len(feature_cols))
    path = config.features_path("train", in_dir)
    held_out = pio.val_ids(args.smoke)
    print(f"Feature version: {feature_version}  |  {len(feature_cols)} features  |  held-out val entities: "
          f"{0 if held_out is None else held_out.len():,}")

    if args.transfer_test:
        print("\n--- France proxy transfer test ---")
        transfer_test(path, in_dir, feature_cols, prior_col, held_out, args, mono)
        return

    print(f"Streaming {path.name}: {args.neg_ratio:g} neg/pos per entity (min {args.min_negs}), "
          f"split train/val/calib = {1 - args.val_frac - args.calib_frac:.0%}/{args.val_frac:.0%}/{args.calib_frac:.0%} by entity hash")
    s = load_slices(path, feature_cols, prior_col, held_out, args.neg_ratio, args.min_negs,
                    args.val_frac, args.calib_frac, args.batch_rows)
    tr, va, ca = s[TRAIN], s[VAL], s[CALIB]
    rss_gb("after streaming")
    for name, d in (("train", tr), ("val", va), ("calib (unsampled)", ca)):
        n = len(d["y"])
        print(f"  {name:18s} {n:>12,} rows  pos rate {d['y'].mean() if n else 0:.4f}")
    if not len(tr["y"]) or not len(va["y"]) or not len(ca["y"]):
        raise SystemExit("A split is empty; too few entities for this --val-frac/--calib-frac.")

    # ── train ────────────────────────────────────────────────────────────────
    params = {**LGB_PARAMS, "monotone_constraints": mono}
    dtrain = lgb.Dataset(tr["X"], label=tr["y"], feature_name=feature_cols)
    dval = lgb.Dataset(va["X"], label=va["y"], reference=dtrain)
    dtrain.construct(); dval.construct()
    rss_gb("after Dataset construct")
    print("Training LightGBM ...", flush=True)
    t_fit = time.perf_counter()
    model = lgb.train(params, dtrain, num_boost_round=args.rounds, valid_sets=[dtrain, dval],
                      valid_names=["train", "val"],
                      callbacks=[lgb.early_stopping(args.early_stop), lgb.log_evaluation(period=25)])
    fit_s = time.perf_counter() - t_fit
    rss_gb("after fit")
    del dtrain, tr
    val_preds = model.predict(va["X"], num_iteration=model.best_iteration)
    val_auc = float(roc_auc_score(va["y"], val_preds))
    print(f"  best_iteration {model.best_iteration}  fit {fit_s:.0f}s  Val AUC {val_auc:.4f}  "
          f"Val Brier {brier_score_loss(va['y'], val_preds):.5f} (sampled distribution)")
    del dval, va

    # ── calibrate on the UNSAMPLED slice ─────────────────────────────────────
    print("Fitting isotonic calibration on the unsampled slice ...")
    raw = _predict(model, ca["X"], model.best_iteration)
    ir, cal = fit_isotonic_with_check(raw, ca["y"], ca["b"])
    print(f"  calib rows {len(raw):,}  prevalence {cal['prevalence']:.4f} ({cal['neg_per_pos']:.2f} neg/pos)")
    print(f"  mean raw prob {cal['mean_raw']:.4f} vs true {cal['prevalence']:.4f}  "
          f"(raw LightGBM output, before isotonic)")
    print(f"  Brier raw {cal['brier_raw']:.5f} -> cross-fitted calibrated {cal['brier_cv']:.5f}")
    print(f"  ECE   raw {cal['ece_raw']:.5f} -> cross-fitted calibrated {cal['ece_cv']:.5f}")
    print_reliability(cal)

    # ── save ─────────────────────────────────────────────────────────────────
    model_out = config.model_path(out_dir)
    model.save_model(str(model_out), num_iteration=model.best_iteration)
    meta = {
        "kind": "lightgbm",
        "feature_version": feature_version,
        "feature_names": list(feature_names),
        "best_iteration": model.best_iteration,
        "val_auc": val_auc,
        "seed": config.SEED,
        "sampling": {"neg_ratio": args.neg_ratio, "min_negs": args.min_negs, "per_entity": True,
                     "entity_subsample": None, "val_frac": args.val_frac, "calib_frac": args.calib_frac,
                     "calibration_slice": "unsampled"},
        "calibration": cal,
        "fit_seconds": fit_s,
    }
    model_out.with_suffix(".meta").write_text(json.dumps(meta, indent=2), encoding=config.ENCODING)
    with open(config.calibrator_path(out_dir), "wb") as f:
        pickle.dump(ir, f)

    print(f"model.txt        -> {model_out}")
    print(f"model.meta       -> {model_out.with_suffix('.meta')}")
    print(f"calibrator.pkl   -> {config.calibrator_path(out_dir)}")
    print(f"s4 done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())

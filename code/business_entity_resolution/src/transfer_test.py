import polars as pl
import lightgbm as lgb
import argparse
from sklearn.metrics import log_loss, roc_auc_score
from features import FEATURE_NAMES

def train_and_eval(X_train, y_train, X_test, y_test, feature_cols, monotone_constraints):
    """Helper to train a basic LightGBM model and evaluate it on a target dataset."""
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'monotone_constraints': monotone_constraints,
        'seed': 42,
        'verbosity': -1
    }
    
    # We train for a fixed number of rounds for a fair baseline comparison
    model = lgb.train(
        params,
        train_data,
        num_boost_round=150
    )
    
    preds = model.predict(X_test)
    loss = log_loss(y_test, preds)
    auc = roc_auc_score(y_test, preds)
    
    return loss, auc

def run_transfer_test(features_path: str, norm_s1_path: str):
    """
    Trains on US data and evaluates on India data, and vice versa.
    This simulates the zero-shot country transfer required for France in the test set.
    """
    print("Loading features and country labels...")
    features_df = pl.read_parquet(features_path)
    
    # The features parquet doesn't have the country (to keep memory low).
    # We join with the normalised S1 file on entity_id to get it.
    norm_s1_df = pl.read_parquet(norm_s1_path).select(["entity_id", "country"])
    
    df = features_df.join(
        norm_s1_df, 
        left_on="source1_entity_id", 
        right_on="entity_id", 
        how="inner"
    )
    
    feature_cols = [f[0] for f in FEATURE_NAMES]
    monotone_constraints = [f[1] for f in FEATURE_NAMES]
    
    print("Filtering datasets by country...")
    # Supporting variations just in case the raw data uses acronyms
    us_df = df.filter(pl.col("country").is_in(["US", "United States", "USA"]))
    in_df = df.filter(pl.col("country").is_in(["IN", "India"]))
    
    print(f"US Entities: {us_df.select('source1_entity_id').n_unique()} | Rows: {len(us_df)}")
    print(f"India Entities: {in_df.select('source1_entity_id').n_unique()} | Rows: {len(in_df)}")
    
    if len(us_df) == 0 or len(in_df) == 0:
        print("Error: Could not find records for one of the countries. Check country names in norm_s1.")
        return

    X_us = us_df.select(feature_cols).to_numpy()
    y_us = us_df.select("label").to_numpy().ravel()
    
    X_in = in_df.select(feature_cols).to_numpy()
    y_in = in_df.select("label").to_numpy().ravel()
    
    print("\n--- Training on US ---")
    loss_us_us, auc_us_us = train_and_eval(X_us, y_us, X_us, y_us, feature_cols, monotone_constraints)
    loss_us_in, auc_us_in = train_and_eval(X_us, y_us, X_in, y_in, feature_cols, monotone_constraints)
    
    print("\n--- Training on India ---")
    loss_in_in, auc_in_in = train_and_eval(X_in, y_in, X_in, y_in, feature_cols, monotone_constraints)
    loss_in_us, auc_in_us = train_and_eval(X_in, y_in, X_us, y_us, feature_cols, monotone_constraints)
    
    print("\n================ TRANSFER DEGRADATION REPORT ================")
    print(f"{'Train':<10} | {'Test':<10} | {'Log Loss':<10} | {'AUC':<10}")
    print("-" * 50)
    print(f"{'US':<10} | {'US (In)':<10} | {loss_us_us:<10.4f} | {auc_us_us:<10.4f}")
    print(f"{'US':<10} | {'India (Out)':<8} | {loss_us_in:<10.4f} | {auc_us_in:<10.4f}")
    print(f"-> Degradation (US -> India): AUC Drop = {auc_us_us - auc_us_in:.4f}")
    print("-" * 50)
    print(f"{'India':<10} | {'India (In)':<10} | {loss_in_in:<10.4f} | {auc_in_in:<10.4f}")
    print(f"{'India':<10} | {'US (Out)':<10} | {loss_in_us:<10.4f} | {auc_in_us:<10.4f}")
    print(f"-> Degradation (India -> US): AUC Drop = {auc_in_in - auc_in_us:.4f}")
    print("=============================================================")
    print("Note: This drop estimates what will happen to the France data in the test set.")
    print("If the drop is severe, you need to rely more heavily on your monotonic constraints and language-agnostic features.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, required=True, help="Path to the featurised dataset parquet")
    parser.add_argument("--norm-s1", type=str, required=True, help="Path to S1 norm parquet (for country labels)")
    args = parser.parse_args()
    
    run_transfer_test(args.features, args.norm_s1)

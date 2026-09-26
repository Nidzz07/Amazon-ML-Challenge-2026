import polars as pl
import lightgbm as lgb
import json
import argparse
from features import FEATURE_NAMES, FEATURE_VERSION

def train_model(train_path: str, val_path: str, model_out_path: str):
    """
    Trains a LightGBM binary classifier on the featurised parquet files.
    Applies monotonic constraints as defined in features.FEATURE_NAMES.
    Uses early stopping on the validation set.
    """
    # FEATURE_NAMES is expected to be a list of tuples: (feature_name, monotonic_direction)
    feature_cols = [f[0] for f in FEATURE_NAMES]
    monotone_constraints = [f[1] for f in FEATURE_NAMES]

    print(f"Loading featurised training data from {train_path}...")
    train_df = pl.read_parquet(train_path)
    val_df = pl.read_parquet(val_path)

    # Convert polars DataFrames to numpy arrays for LightGBM
    X_train = train_df.select(feature_cols).to_numpy()
    y_train = train_df.select("label").to_numpy().ravel()
    
    X_val = val_df.select(feature_cols).to_numpy()
    y_val = val_df.select("label").to_numpy().ravel()

    print("Creating LightGBM datasets...")
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=train_data)

    # Initial hyperparameters as specified in the roadmap
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'monotone_constraints': monotone_constraints,
        'seed': 42,
        'verbosity': -1
    }

    print("Training LightGBM model...")
    # Use early stopping against our held-out validation set
    callbacks = [
        lgb.early_stopping(stopping_rounds=50), 
        lgb.log_evaluation(period=10)
    ]
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=callbacks
    )

    print(f"Saving model to {model_out_path}...")
    model.save_model(model_out_path)
    
    # The roadmap explicitly warns about feature mismatch. 
    # We save the FEATURE_VERSION in a metadata file next to the model.
    # The scoring script will check this to prevent silent garbage predictions.
    with open(model_out_path + ".meta", "w") as f:
        json.dump({"FEATURE_VERSION": FEATURE_VERSION}, f)
        
    print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True, help="Path to featurised train parquet")
    parser.add_argument("--val", type=str, required=True, help="Path to featurised val parquet")
    parser.add_argument("--out-model", type=str, required=True, help="Output path for model.txt")
    args = parser.parse_args()
    
    train_model(args.train, args.val, args.out_model)

import polars as pl
import lightgbm as lgb
import numpy as np
import pickle
import argparse
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
from features import FEATURE_NAMES

def calibrate_model(model_path: str, calib_path: str, out_calibrator: str, out_plot: str):
    """
    Fits an IsotonicRegression calibrator on a disjoint calibration slice.
    Reports the Brier score and generates a Reliability Curve.
    """
    print(f"Loading LightGBM model from {model_path}...")
    model = lgb.Booster(model_file=model_path)
    
    feature_cols = [f[0] for f in FEATURE_NAMES]
    
    print(f"Loading calibration data from {calib_path}...")
    calib_df = pl.read_parquet(calib_path)
    
    X_calib = calib_df.select(feature_cols).to_numpy()
    y_calib = calib_df.select("label").to_numpy().ravel()
    
    print("Predicting uncalibrated probabilities...")
    # Predict raw probabilities from LGBM
    uncalibrated_probs = model.predict(X_calib)
    
    print("Fitting Isotonic Regression...")
    ir = IsotonicRegression(out_of_bounds='clip')
    calibrated_probs = ir.fit_transform(uncalibrated_probs, y_calib)
    
    print(f"Saving calibrator to {out_calibrator}...")
    with open(out_calibrator, 'wb') as f:
        pickle.dump(ir, f)
        
    # Calculate Brier Scores
    brier_uncalibrated = brier_score_loss(y_calib, uncalibrated_probs)
    brier_calibrated = brier_score_loss(y_calib, calibrated_probs)
    
    print("\n" + "=" * 40)
    print(f"Brier Score (Uncalibrated): {brier_uncalibrated:.5f}")
    print(f"Brier Score (Calibrated):   {brier_calibrated:.5f}")
    print("=" * 40 + "\n")
    
    # Calculate Reliability Curve points
    prob_true_unc, prob_pred_unc = calibration_curve(y_calib, uncalibrated_probs, n_bins=10)
    prob_true_cal, prob_pred_cal = calibration_curve(y_calib, calibrated_probs, n_bins=10)
    
    print("Reliability Curve Data (Calibrated):")
    print(f"{'Mean Predicted Prob':<25} | {'Fraction of Positives (Actual)'}")
    print("-" * 55)
    for p_pred, p_true in zip(prob_pred_cal, prob_true_cal):
        print(f"{p_pred:<25.4f} | {p_true:.4f}")
    print("\n")
    
    # Try to plot if matplotlib is installed
    try:
        import matplotlib.pyplot as plt
        print(f"Generating reliability curve plot at {out_plot}...")
        
        plt.figure(figsize=(8, 8))
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated', color='black')
        plt.plot(prob_pred_unc, prob_true_unc, marker='s', label='Uncalibrated LightGBM')
        plt.plot(prob_pred_cal, prob_true_cal, marker='o', label='Isotonic Calibration')
        
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title('Reliability Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(out_plot)
        plt.close()
    except ImportError:
        print("matplotlib not installed. Skipping the visual plot generation. You can install it via 'pip install matplotlib'.")

    print("Calibration complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained model.txt")
    parser.add_argument("--calib-data", type=str, required=True, help="Path to calibration split parquet")
    parser.add_argument("--out-calibrator", type=str, required=True, help="Output path for calibrator.pkl")
    parser.add_argument("--out-plot", type=str, default="reliability_curve.png", help="Output path for the plot")
    args = parser.parse_args()
    
    calibrate_model(args.model, args.calib_data, args.out_calibrator, args.out_plot)

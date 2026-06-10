import argparse
import json
import subprocess
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "gld_price_data.csv"
MODEL_PATH = ROOT_DIR / "model" / "model.pkl"
OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def ensure_model_exists() -> None:
    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}. Training a new model now...")
        subprocess.run([sys.executable, str(ROOT_DIR / "src" / "train.py")], check=True)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def plot_price_trend(df: pd.DataFrame, last_date, actual, prediction):
    plt.figure(figsize=(10, 6))
    plt.plot(df["Date"], df["GLD"], label="Actual GLD", color="tab:blue")
    plt.scatter([last_date], [actual], label="Last actual price", color="tab:orange", zorder=5)
    plt.scatter([last_date], [prediction], label="Predicted price", color="tab:red", marker="X", s=100, zorder=5)
    plt.title("Gold Price Trend and Prediction")
    plt.xlabel("Date")
    plt.ylabel("GLD")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "gold_price_prediction.png"
    plt.savefig(plot_path)
    plt.close()
    return plot_path


def plot_actual_vs_predicted(y_test, y_pred_test):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred_test, alpha=0.6, color="tab:green", edgecolor="k")
    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect prediction")
    plt.title("Actual vs Predicted Gold Price on Test Set")
    plt.xlabel("Actual GLD")
    plt.ylabel("Predicted GLD")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "actual_vs_predicted_test_set.png"
    plt.savefig(plot_path)
    plt.close()
    return plot_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict gold price with a trained model.")
    parser.add_argument("--spx", type=float, help="SPX feature value.")
    parser.add_argument("--uso", type=float, help="USO feature value.")
    parser.add_argument("--slv", type=float, help="SLV feature value.")
    parser.add_argument("--eurusd", type=float, help="EUR/USD feature value.")
    parser.add_argument("--latest", action="store_true", help="Use the latest dataset row for prediction.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation graphs and only make a prediction.")
    return parser.parse_args()


def build_sample_from_args(args: argparse.Namespace, df: pd.DataFrame) -> pd.DataFrame:
    supplied = [args.spx, args.uso, args.slv, args.eurusd]
    if any(value is not None for value in supplied):
        if not all(value is not None for value in supplied):
            raise ValueError("If providing custom features, all of --spx --uso --slv --eurusd must be set.")
        return pd.DataFrame(
            [[args.spx, args.uso, args.slv, args.eurusd]],
            columns=["SPX", "USO", "SLV", "EUR/USD"],
        )
    return df.drop(columns=["Date", "GLD"]).iloc[[-1]]


def main():
    args = parse_args()
    ensure_model_exists()

    model = joblib.load(MODEL_PATH)
    df = load_data()

    sample = build_sample_from_args(args, df)
    actual = df["GLD"].iloc[-1] if len(sample) == 1 and sample.index[0] == len(df) - 1 else None
    last_date = df["Date"].iloc[-1]
    prediction = model.predict(sample)[0]

    print("Prediction input:")
    print(json.dumps(sample.to_dict(orient="records")[0], indent=2))
    if actual is not None:
        print("Actual Gold Price:", actual)
        print("Prediction error:", abs(prediction - actual))
    print("Predicted Gold Price:", prediction)

    if not args.skip_eval:
        trend_path = plot_price_trend(df, last_date, actual if actual is not None else prediction, prediction)
        print("Saved prediction graph at:", trend_path)

        split_index = int(len(df) * 0.8)
        feature_df = df.drop(columns=["Date", "GLD"])
        label_df = df["GLD"]
        X_test = feature_df.iloc[split_index:]
        y_test = label_df.iloc[split_index:]
        y_pred_test = model.predict(X_test)

        eval_path = plot_actual_vs_predicted(y_test, y_pred_test)
        print("Saved actual vs predicted graph at:", eval_path)

        test_metrics = {
            "prediction_date": last_date.isoformat(),
            "input_features": sample.to_dict(orient="records")[0],
            "predicted_value": float(prediction),
            "actual_value": float(actual) if actual is not None else None,
            "prediction_error": float(abs(prediction - actual)) if actual is not None else None,
            "model_report": "Actual vs predicted test set chart saved.",
        }
        with open(OUTPUT_DIR / "prediction_summary.json", "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2)
        print("Saved prediction summary at:", OUTPUT_DIR / "prediction_summary.json")


if __name__ == "__main__":
    main()
 
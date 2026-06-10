import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "gld_price_data.csv"
MODEL_DIR = ROOT_DIR / "model"
OUTPUT_DIR = ROOT_DIR / "output"
MODEL_PATH = MODEL_DIR / "model.pkl"
TRAINING_METRICS_PATH = OUTPUT_DIR / "training_metrics.json"
MODEL_COMPARISON_PATH = OUTPUT_DIR / "model_comparison.json"
FEATURE_IMPORTANCE_PLOT = OUTPUT_DIR / "feature_importance.png"
COMPARISON_PLOT = OUTPUT_DIR / "model_comparison.png"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def split_time_series(df: pd.DataFrame, train_fraction: float = 0.8):
    split_index = int(len(df) * train_fraction)
    X = df.drop(columns=["Date", "GLD"])
    y = df["GLD"]
    return X.iloc[:split_index], X.iloc[split_index:], y.iloc[:split_index], y.iloc[split_index:]


def build_models() -> dict:
    return {
        "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=200, max_depth=8, random_state=42),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=200, max_depth=10, random_state=42),
        "LinearRegression": LinearRegression(),
    }


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[dict, np.ndarray]:
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    metrics = {
        "r2": r2_score(y_test, predictions),
        "mae": mean_absolute_error(y_test, predictions),
        "mse": mse,
        "rmse": np.sqrt(mse),
        "test_rows": len(X_test),
    }
    return metrics, predictions


def select_best_model(comparison: dict) -> str:
    ordered = sorted(comparison.items(), key=lambda item: (item[1]["rmse"], -item[1]["r2"]))
    return ordered[0][0]


def save_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_)
    else:
        print("Selected model does not support feature importance plotting.")
        return

    indices = np.argsort(values)[::-1]
    plt.figure(figsize=(8, 5))
    plt.bar([feature_names[i] for i in indices], values[indices], color="tab:blue")
    plt.title("Feature Importance")
    plt.ylabel("Importance")
    plt.xlabel("Feature")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PLOT)
    plt.close()


def save_model_comparison_plot(comparison: dict):
    names = list(comparison.keys())
    rmses = [comparison[name]["rmse"] for name in names]
    r2s = [comparison[name]["r2"] for name in names]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(names, rmses, color="tab:blue", alpha=0.7, label="RMSE")
    ax1.set_ylabel("RMSE", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_title("Model Comparison")

    ax2 = ax1.twinx()
    ax2.plot(names, r2s, color="tab:red", marker="o", label="R2")
    ax2.set_ylabel("R2", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    fig.tight_layout()
    fig.legend(loc="upper left", bbox_to_anchor=(0.15, 0.85))
    plt.savefig(COMPARISON_PLOT)
    plt.close()


def save_comparison(comparison: dict):
    serializable = {
        name: {
            "r2": float(metrics["r2"]),
            "mae": float(metrics["mae"]),
            "mse": float(metrics["mse"]),
            "rmse": float(metrics["rmse"]),
        }
        for name, metrics in comparison.items()
    }
    with open(MODEL_COMPARISON_PATH, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


def main():
    df = load_data()
    X_train, X_test, y_train, y_test = split_time_series(df)
    print(f"Training on {len(X_train)} rows and evaluating on {len(X_test)} rows.")

    models = build_models()
    comparison = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        metrics, _ = evaluate_model(model, X_test, y_test)
        comparison[name] = metrics
        print(f"  {name} metrics: R2={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}")

    best_name = select_best_model(comparison)
    best_model = models[best_name]
    best_metrics = comparison[best_name]
    print(f"Best model: {best_name}")

    joblib.dump(best_model, MODEL_PATH)
    print(f"Saved best model ({best_name}) to {MODEL_PATH}")

    training_payload = {
        "best_model": best_name,
        "best_metrics": {k: float(v) for k, v in best_metrics.items()},
        "model_comparison": {
            name: {"r2": float(metrics["r2"]), "mae": float(metrics["mae"]), "mse": float(metrics["mse"]), "rmse": float(metrics["rmse"])}
            for name, metrics in comparison.items()
        },
    }
    with open(TRAINING_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(training_payload, f, indent=2)

    save_comparison(comparison)
    save_model_comparison_plot(comparison)
    save_feature_importance(best_model, X_train.columns)

    print("Model comparison saved to", MODEL_COMPARISON_PATH)
    print("Model comparison plot saved to", COMPARISON_PLOT)
    print("Feature importance plot saved to", FEATURE_IMPORTANCE_PLOT)
    print("Training metrics saved to", TRAINING_METRICS_PATH)

    print("Final model metrics:")
    for key, value in best_metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
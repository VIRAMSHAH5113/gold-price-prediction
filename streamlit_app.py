import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data" / "gld_price_data.csv"
MODEL_PATH = ROOT_DIR / "model" / "model.pkl"
TRAINING_METRICS_PATH = ROOT_DIR / "output" / "training_metrics.json"
MODEL_COMPARISON_PATH = ROOT_DIR / "output" / "model_comparison.json"


def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def load_model():
    return joblib.load(MODEL_PATH)


def load_training_metrics():
    if TRAINING_METRICS_PATH.exists():
        with open(TRAINING_METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_comparison():
    if MODEL_COMPARISON_PATH.exists():
        with open(MODEL_COMPARISON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def plot_time_series(df, prediction, prediction_date):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["Date"], df["GLD"], label="Gold Price", color="tab:blue")
    ax.scatter([prediction_date], [prediction], color="tab:red", label="Prediction", zorder=5)
    ax.set_title("Gold Price History")
    ax.set_xlabel("Date")
    ax.set_ylabel("GLD")
    ax.legend()
    ax.grid(True)
    return fig


def plot_model_comparison(comparison: dict):
    if not comparison:
        return None
    df = pd.DataFrame(comparison).T
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df.index, df["rmse"], color="tab:blue", alpha=0.8)
    ax.set_ylabel("RMSE")
    ax.set_title("Model RMSE Comparison")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    return fig


def plot_feature_importance():
    feature_plot = ROOT_DIR / "output" / "feature_importance.png"
    return feature_plot if feature_plot.exists() else None


def main():
    st.set_page_config(page_title="Gold Price Prediction Dashboard", layout="wide")
    st.title("Gold Price Prediction Dashboard")

    df = load_data()
    model_metrics = load_training_metrics()
    comparison = load_comparison()

    st.sidebar.header("Prediction Input")
    spx = st.sidebar.number_input("SPX", value=float(df["SPX"].iloc[-1]), format="%.4f")
    uso = st.sidebar.number_input("USO", value=float(df["USO"].iloc[-1]), format="%.4f")
    slv = st.sidebar.number_input("SLV", value=float(df["SLV"].iloc[-1]), format="%.4f")
    eurusd = st.sidebar.number_input("EUR/USD", value=float(df["EUR/USD"].iloc[-1]), format="%.4f")
    prediction = None
    if st.sidebar.button("Predict"):
        model = load_model()
        sample = pd.DataFrame([[spx, uso, slv, eurusd]], columns=["SPX", "USO", "SLV", "EUR/USD"])
        prediction = model.predict(sample)[0]
        st.sidebar.success(f"Predicted Gold Price: {prediction:.4f}")
        st.sidebar.write("**Input values:**")
        st.sidebar.json(sample.to_dict(orient="records")[0])

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Gold Price Trend")
        latest_date = df["Date"].iloc[-1]
        latest_price = df["GLD"].iloc[-1]
        model = load_model()
        last_sample = df.drop(columns=["Date", "GLD"]).iloc[[-1]]
        last_prediction = model.predict(last_sample)[0]
        fig = plot_time_series(df, last_prediction, latest_date)
        st.pyplot(fig)
        st.markdown(f"**Latest actual price:** {latest_price:.4f}")
        st.markdown(f"**Latest predicted price:** {last_prediction:.4f}")
        if prediction is not None:
            st.markdown(f"**Custom prediction:** {prediction:.4f}")

    with col2:
        st.subheader("Model Summary")
        if model_metrics:
            best_metrics = model_metrics.get("best_metrics", {})
            st.metric("Best model", model_metrics.get("best_model", "Unknown"))
            metric_cols = st.columns(3)
            metric_cols[0].metric("R2", f"{best_metrics.get('r2', 0):.4f}")
            metric_cols[1].metric("MAE", f"{best_metrics.get('mae', 0):.4f}")
            metric_cols[2].metric("RMSE", f"{best_metrics.get('rmse', 0):.4f}")
            st.markdown("**Model comparison table**")
            st.table(pd.DataFrame(comparison).T)
        else:
            st.warning("Training metrics not found. Run src/train.py first.")

        st.subheader("Comparison Chart")
        comparison_fig = plot_model_comparison(comparison)
        if comparison_fig is not None:
            st.pyplot(comparison_fig)
        elif comparison:
            st.info("Model comparison image unavailable.")

        st.subheader("Feature Importance")
        feature_plot = plot_feature_importance()
        if feature_plot is not None:
            st.image(feature_plot, use_column_width=True)
        else:
            st.info("Feature importance plot not available yet. Run src/train.py to generate it.")

    st.subheader("Raw Dataset Preview")
    st.dataframe(df.tail(10))

    if not comparison:
        st.warning("No model comparison data found. Train the models with src/train.py first.")


if __name__ == "__main__":
    main()

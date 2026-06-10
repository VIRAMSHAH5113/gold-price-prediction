# Gold Price Prediction

This project trains a Random Forest model to predict gold prices (`GLD`) using features from the dataset.

## Run the project

Train the model and generate graphs:

```bash
python main.py
```

Force retraining and regenerate graphs:

```bash
python main.py --retrain
```

Train using the full comparison workflow and regenerate feature importance and model comparison output:

```bash
python src/train.py
```

Run the Streamlit dashboard:

```bash
streamlit run streamlit_app.py
```

## Predict with custom input

The prediction script can also accept manual feature values:

```bash
python src/predict.py --spx 1500 --uso 80 --slv 16 --eurusd 1.2
```

If no custom values are provided, it uses the latest row from `data/gld_price_data.csv`.

## Output files

- `output/gold_price_prediction.png` — gold price trend with the latest prediction.
- `output/actual_vs_predicted_test_set.png` — actual vs predicted values on the test set.
- `output/prediction_summary.json` — prediction summary for the latest prediction.
- `output/feature_importance.png` — feature importance from model training.
- `output/model_comparison.json` — performance comparison for all trained models.
- `output/training_metrics.json` — evaluation metrics from training.

# Gold Price Prediction Dashboard

## Live Demo

https://gold-price-prediction-bgmxvxqcvsgezihjun3cai.streamlit.app/

## GitHub Repository

https://github.com/VIRAMSHAH5113/gold-price-prediction

## Project Overview

This project uses Machine Learning to predict Gold Prices (GLD) based on financial indicators such as:

* SPX
* USO
* SLV
* EUR/USD

The application includes a Streamlit dashboard where users can enter feature values and obtain predicted gold prices.

## Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Joblib

## Run the Project

Train the model and generate graphs:

```bash
python main.py
```

Force retraining and regenerate graphs:

```bash
python main.py --retrain
```

Train using the full comparison workflow:

```bash
python src/train.py
```

Run the Streamlit dashboard:

```bash
streamlit run streamlit_app.py
```

## Predict with Custom Input

```bash
python src/predict.py --spx 1500 --uso 80 --slv 16 --eurusd 1.2
```

If no custom values are provided, the latest row from `data/gld_price_data.csv` is used.

## Output Files

* `output/gold_price_prediction.png`
* `output/actual_vs_predicted_test_set.png`
* `output/prediction_summary.json`
* `output/feature_importance.png`
* `output/model_comparison.json`
* `output/training_metrics.json`

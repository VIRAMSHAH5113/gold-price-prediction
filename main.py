import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "model" / "model.pkl"

parser = argparse.ArgumentParser(description="Train the gold price model and generate prediction graphs.")
parser.add_argument("--retrain", action="store_true", help="Force retrain the model before prediction.")
args = parser.parse_args()

if args.retrain or not MODEL_PATH.exists():
    print("Training model...")
    subprocess.run([sys.executable, str(PROJECT_ROOT / "src" / "train.py")], check=True)
else:
    print("Model exists. Skipping training unless --retrain is provided.")

print("\nPredicting sample value...")
subprocess.run([sys.executable, str(PROJECT_ROOT / "src" / "predict.py")], check=True)

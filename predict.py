"""
Inference outside the training notebooks.

Loads a saved .joblib model and runs predict() on already-processed
feature rows (same columns as features.train/test.csv).

Usage:
  py -3 -m pip install -r requirements.txt
  py -3 predict.py --model rf
  py -3 predict.py --model ann --input artificial_neural_network/data/features.test.csv --n 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent

MODELS = {
    "rf": ROOT / "random_forests" / "model" / "model_rf.joblib",
    "ann": ROOT / "artificial_neural_network" / "model" / "model_ann.joblib",
    "knn": ROOT / "k_nearest_neighbor_4techniques" / "model" / "model_Knn.joblib",
}

DEFAULT_FEATURES = {
    "rf": ROOT / "random_forests" / "data" / "features.test.csv",
    "ann": ROOT / "artificial_neural_network" / "data" / "features.test.csv",
    "knn": ROOT / "k_nearest_neighbor_4techniques" / "data" / "features.test.csv",
}

DROP_COLS = {"id", "label"}


def build_matrix(df: pd.DataFrame, model) -> pd.DataFrame:
    """Keep only feature columns and align order to the trained model."""
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None:
        # Match train columns; fill missing one-hot cols with 0
        X = X.reindex(columns=list(feature_names), fill_value=0.0)

    return X


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict high-salary label from a saved joblib model")
    parser.add_argument("--model", choices=sorted(MODELS), default="rf", help="Which saved model to load")
    parser.add_argument("--input", type=Path, default=None, help="Processed features CSV (default: that model's test features)")
    parser.add_argument("--n", type=int, default=5, help="Number of rows to predict (default: 5)")
    args = parser.parse_args()

    model_path = MODELS[args.model]
    input_path = args.input or DEFAULT_FEATURES[args.model]

    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}\nTip: run `git lfs pull` after cloning.")
    if not input_path.exists():
        raise SystemExit(f"Features CSV not found: {input_path}")

    model = joblib.load(model_path)
    df = pd.read_csv(input_path).head(args.n)
    X = build_matrix(df, model)

    preds = model.predict(X)
    out = pd.DataFrame({"prediction": preds})
    if "id" in df.columns:
        out.insert(0, "id", df["id"].to_numpy())
    if "label" in df.columns:
        out["true_label"] = df["label"].to_numpy()

    print(f"Loaded: {model_path.relative_to(ROOT)}")
    print(f"Rows:   {len(out)} from {input_path.relative_to(ROOT)}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

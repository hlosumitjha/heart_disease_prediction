"""
Utility to load the trained pipeline and make predictions.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union

from utils import load_pickle

DEFAULT_MODEL_PATH = os.path.join("model", "heart_model.pkl")


def load_model(model_path: str = DEFAULT_MODEL_PATH):
    return load_pickle(model_path)


def predict_single(model, features: Dict[str, Union[int, float, str]]) -> Tuple[int, float]:
    """
    features: dict mapping feature_name -> value (numeric or categorical)
    returns: (predicted_class, probability_of_heart_disease)
    """

    # Ensure correct column order
    feat_order = list(getattr(model, "feature_names_in_", []))
    if not feat_order:
        raise ValueError("Model missing feature_names_in_. Retrain with newer sklearn.")

    # Convert to dataframe matching training shape
    row_df = pd.DataFrame([features], columns=feat_order)

    # Predict
    proba = model.predict_proba(row_df)[0][1]
    pred = int(proba >= 0.5)

    return pred, float(proba)


def predict_batch(model, df: pd.DataFrame) -> pd.DataFrame:
    """Return df with added columns: prediction, probability"""

    feat_order = list(getattr(model, "feature_names_in_", []))

    missing = [c for c in feat_order if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in input: {missing}")

    df_ordered = df[feat_order].copy()

    preds = model.predict(df_ordered)
    probas = model.predict_proba(df_ordered)[:, 1]

    out = df.copy()
    out["prediction"] = preds
    out["probability"] = probas

    return out


if __name__ == "__main__":
    # Example usage
    model = load_model()
    feature_names = list(getattr(model, "feature_names_in_", []))

    example = {name: 0 for name in feature_names}
    pred, proba = predict_single(model, example)
    print("Prediction:", "Heart Disease" if pred == 1 else "No Heart Disease")
    print("Probability:", round(proba, 4))

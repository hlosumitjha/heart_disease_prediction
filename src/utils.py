import os
import pickle
import pandas as pd
from typing import Tuple

def load_csv(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Loaded CSV is empty.")
    return df

def split_features_target(df: pd.DataFrame, target_col: str = "target") -> Tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise KeyError(
            f"Target column '{target_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def save_pickle(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pickle file not found at: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

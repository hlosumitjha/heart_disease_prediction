"""
Train a Logistic Regression model for Heart Disease Prediction.
Saves a full preprocessing+model pipeline to model/heart_model.pkl
"""

import os
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)

from utils import load_csv, split_features_target, save_pickle

DEFAULT_CSV = os.path.join("data", "HeartDiseaseTrain-Test.csv")
DEFAULT_MODEL_PATH = os.path.join("model", "heart_model.pkl")


def build_pipeline(numeric_features, categorical_features):

    numeric_preprocess = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_preprocess = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_preprocess, numeric_features),
            ("cat", categorical_preprocess, categorical_features),
        ]
    )

    model = LogisticRegression(max_iter=1000, solver="lbfgs")

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("clf", model)
    ])
    
    return pipe


def main(csv_path: str = DEFAULT_CSV, model_path: str = DEFAULT_MODEL_PATH,
         test_size: float = 0.2, random_state: int = 42):

    print(f"Loading data from: {csv_path}")
    df = load_csv(csv_path).copy()

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    after = len(df)
    if after < before:
        print(f"Dropped {before - after} duplicate rows.")

    # Split into X and y
    X, y = split_features_target(df, target_col="HeartDisease")

    # Correct feature separation
    numeric_features = ["Age", "RestingBP", "Cholesterol", "FastingBS",
                        "MaxHR", "Oldpeak"]

    categorical_features = ["Sex", "ChestPainType", "RestingECG",
                            "ExerciseAngina", "ST_Slope"]

    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")
    print(f"Target distribution:\n{y.value_counts(dropna=False)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipe = build_pipeline(numeric_features, categorical_features)

    print("Training Logistic Regression model...")
    pipe.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    print("\nResults:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print(f"Saving model to: {model_path}")
    save_pickle(pipe, model_path)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV, help="Path to dataset CSV")
    parser.add_argument("--out", type=str, default=DEFAULT_MODEL_PATH, help="Output pickle path")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(csv_path=args.csv, model_path=args.out, test_size=args.test_size, random_state=args.seed)

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Tuple, List

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

MODEL_PATH = Path("xgb_credit_model.pkl")

def get_feature_config(df: pd.DataFrame, target_col: str = "Risk") -> Dict[str, str]:
    """
    Gibt für jede Feature-Spalte an, ob sie 'numeric' oder 'categorical' ist.
    """
    features = [c for c in df.columns if c != target_col]
    cfg = {}
    for c in features:
        if pd.api.types.is_numeric_dtype(df[c]):
            cfg[c] = "numeric"
        else:
            cfg[c] = "categorical"
    return cfg

def encode_fit_transform(df: pd.DataFrame, target_col: str = "Risk"):
    cfg = get_feature_config(df, target_col)
    df_model = df.copy()

    # Zielvariable encoden
    le_target = LabelEncoder()
    df_model[target_col] = le_target.fit_transform(df_model[target_col])

    # Kategorische Spalten encoden wie im Notebook
    encoders: Dict[str, LabelEncoder] = {}
    for col, kind in cfg.items():
        if kind == "categorical":
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col])
            encoders[col] = le
            joblib.dump(le, f"{col}_encoder.pkl")
    joblib.dump(le_target, "target_encoder.pkl")

    return df_model, encoders, le_target, list(cfg.keys())

def encode_features_for_inference(df_input: pd.DataFrame, encoders: Dict[str, LabelEncoder], feature_order: List[str]) -> pd.DataFrame:
    X = df_input.copy()
    for col in feature_order:
        if col in encoders:
            le = encoders[col]
            # Unbekannte Kategorien robust handhaben: unbekannt -> erste Klasse
            X[col] = X[col].map(lambda v: v if v in le.classes_ else le.classes_[0])
            X[col] = le.transform(X[col])
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
    return X[feature_order]

def train_xgb(df: pd.DataFrame, target_col: str = "Risk"):
    df_model, encoders, le_target, features = encode_fit_transform(df, target_col)

    X = df_model.drop(columns=[target_col])
    y = df_model[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Parametergrid ähnlich wie im Notebook
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2],
        "subsample": [0.7, 1.0],
        "colsample_bytree": [0.7, 1.0],
        "reg_lambda": [1.0, 2.0]
    }

    model = XGBClassifier(
        random_state=42,
        n_jobs=-1,
        objective="binary:logistic" if len(np.unique(y)) == 2 else "multi:softprob",
        eval_metric="logloss"
    )

    grid = GridSearchCV(model, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Artefakte speichern
    joblib.dump(best_model, MODEL_PATH)
    meta = {"encoders": encoders, "target_encoder": le_target, "features": features}
    joblib.dump(meta, MODEL_PATH.with_suffix(".meta.pkl"))

    return best_model, encoders, le_target, features, grid.best_params_, acc

def load_or_train_model(df: pd.DataFrame, target_col: str = "Risk"):
    try:
        model = joblib.load(MODEL_PATH)
        meta = joblib.load(MODEL_PATH.with_suffix(".meta.pkl"))
        return (
            model,
            meta["encoders"],
            meta["target_encoder"],
            meta["features"],
            getattr(model, "get_xgb_params", lambda: {})(),
            np.nan  # Accuracy unbekannt, wenn nur geladen
        )
    except Exception:
        return train_xgb(df, target_col)
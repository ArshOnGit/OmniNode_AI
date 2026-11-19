"""common.py - shared paths and data loading for all model_training scripts."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TRAINED = ROOT / "models" / "trained"
TFLITE = ROOT / "models" / "tflite"
METADATA = ROOT / "models" / "metadata"
RESULTS = ROOT / "results"
FIRMWARE = ROOT / "firmware"

for _p in (TRAINED, TFLITE, METADATA, RESULTS, FIRMWARE / "model_headers"):
    _p.mkdir(parents=True, exist_ok=True)

RISK_NAMES = ["low", "moderate", "high"]


def load_scaler() -> dict:
    return json.loads((PROCESSED / "scaler.json").read_text())


def feature_arrays(scaler: dict):
    feats = scaler["features"]
    mean = np.array([scaler["feature_stats"][f]["mean"] for f in feats], dtype=np.float32)
    std = np.array([scaler["feature_stats"][f]["std"] for f in feats], dtype=np.float32)
    return mean, std


def load_split(name: str, scaler: dict):
    """name = 'training' or 'test'. Returns normalised X, raw target temp (C), risk class."""
    df = pd.read_csv(PROCESSED / f"{name}_dataset.csv")
    cols = [c for c in df.columns if "_lag" in c]
    n_feat = len(scaler["features"])
    mean, std = feature_arrays(scaler)
    X = df[cols].to_numpy(np.float32).reshape(len(df), scaler["window"], n_feat)
    X = ((X - mean) / std).reshape(len(df), -1).astype(np.float32)
    return X, df["target_temp_c"].to_numpy(np.float32), df["risk_class"].to_numpy(np.int32)

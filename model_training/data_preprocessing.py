"""
data_preprocessing.py
Raw greenhouse CSVs -> 5-minute features -> windowed train/test sets.
Run from the repo root:  python model_training/data_preprocessing.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# ---------------- 1. SETTINGS (edit the three column names) ----------------
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "commercial_tomato_greenhouse"
OUT_DIR = ROOT / "data" / "processed"

TIME_COL = "timestamp"      # <-- change to your time column
TEMP_COL = "temperature"    # <-- change to your temperature column (deg C)
RH_COL = "humidity"         # <-- change to your humidity column (%)
TIME_FORMAT = None          # e.g. "%d/%m/%Y %H:%M"; None lets pandas guess

STEP_MIN = 5                # resampling interval in minutes
WINDOW = 15                 # input samples (15 x 5 min = 75 min of history)
HORIZON = 6                 # forecast steps ahead (6 x 5 min = 30 min)
MAX_GAP_STEPS = 3           # fill gaps up to 15 min, longer gaps split the data
TRAIN_FRAC = 0.8

FEATURES = ["temp_c", "rh", "dew_point_c", "dpd_c", "vpd_kpa"]

# Magnus-Tetens constants (the firmware must use the same values)
MT_A, MT_B = 17.27, 237.7


# ---------------- 2. LOAD ----------------
def load_raw() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("year_*/*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {RAW_DIR}/year_*/")
    frames = []
    for f in files:
        df = pd.read_csv(f)
        df = df[[TIME_COL, TEMP_COL, RH_COL]].copy()
        df.columns = ["time", "temp_c", "rh"]
        df["time"] = pd.to_datetime(df["time"], format=TIME_FORMAT, errors="coerce")
        df["temp_c"] = pd.to_numeric(df["temp_c"], errors="coerce")
        df["rh"] = pd.to_numeric(df["rh"], errors="coerce")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["time"]).sort_values("time").drop_duplicates("time")
    print(f"Loaded {len(df):,} raw rows from {len(files)} files")
    return df.set_index("time")


# ---------------- 3. CLEAN + RESAMPLE ----------------
def clean_and_resample(df: pd.DataFrame) -> pd.DataFrame:
    bad = ~(df["temp_c"].between(-10, 60) & df["rh"].between(1, 100))
    df.loc[bad, ["temp_c", "rh"]] = np.nan          # impossible readings -> missing
    df = df.resample(f"{STEP_MIN}min").mean()
    df = df.interpolate(limit=MAX_GAP_STEPS, limit_area="inside")
    df = df.dropna()
    print(f"After cleaning/resampling: {len(df):,} rows at {STEP_MIN}-minute steps")
    return df


# ---------------- 4. FEATURES + RISK LABEL ----------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    t, rh = df["temp_c"], df["rh"]
    gamma = MT_A * t / (MT_B + t) + np.log(rh / 100.0)
    df["dew_point_c"] = MT_B * gamma / (MT_A - gamma)
    df["dpd_c"] = t - df["dew_point_c"]                       # dew point delta
    es = 0.6108 * np.exp(17.27 * t / (t + 237.3))             # saturation vapour pressure, kPa
    df["vpd_kpa"] = es * (1 - rh / 100.0)                     # vapour pressure deficit, kPa
    return df


def add_risk_label(df: pd.DataFrame) -> pd.DataFrame:
    """0 = low, 1 = moderate, 2 = high Botrytis-favourable conditions.
    Heuristic thresholds: environmental indicators only, not proof of infection."""
    high = (df["dpd_c"] <= 1.0) | (df["rh"] >= 90)
    moderate = (df["dpd_c"] <= 2.5) | (df["rh"] >= 85) | (df["vpd_kpa"] < 0.4)
    df["risk_class"] = np.where(high, 2, np.where(moderate, 1, 0))
    return df


# ---------------- 5. WINDOWS ----------------
def build_windows(df: pd.DataFrame) -> pd.DataFrame:
    col_names = [f"{f}_lag{WINDOW - 1 - k}" for k in range(WINDOW) for f in FEATURES]
    step = pd.Timedelta(minutes=STEP_MIN)
    segment_id = (df.index.to_series().diff() != step).cumsum().to_numpy()
    parts = []
    for _, g in df.groupby(segment_id):            # windows never cross a gap
        L = len(g) - WINDOW - HORIZON + 1
        if L <= 0:
            continue
        X = g[FEATURES].to_numpy()
        wins = sliding_window_view(X, (WINDOW, len(FEATURES)))[:L, 0]   # (L, WINDOW, F)
        part = pd.DataFrame(wins.reshape(L, -1), columns=col_names)
        part.insert(0, "timestamp", g.index[WINDOW - 1: WINDOW - 1 + L])
        part["target_temp_c"] = g["temp_c"].to_numpy()[WINDOW - 1 + HORIZON: WINDOW - 1 + HORIZON + L]
        part["risk_class"] = g["risk_class"].to_numpy()[WINDOW - 1: WINDOW - 1 + L]
        parts.append(part)
    return pd.concat(parts, ignore_index=True), col_names


# ---------------- 6. MAIN ----------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = clean_and_resample(load_raw())
    df = add_risk_label(add_features(df))
    df.to_csv(OUT_DIR / "greenhouse_features_5min.csv", index_label="timestamp")

    windows, col_names = build_windows(df)
    split = int(len(windows) * TRAIN_FRAC)
    train = windows.iloc[:split]
    test = windows.iloc[split + WINDOW + HORIZON:]   # gap so no raw row is shared
    train.to_csv(OUT_DIR / "training_dataset.csv", index=False)
    test.to_csv(OUT_DIR / "test_dataset.csv", index=False)

    # normalisation numbers (training data only), reused later by the firmware
    stats = {}
    for f in FEATURES:
        cols = [c for c in col_names if c.startswith(f + "_lag")]
        vals = train[cols].to_numpy()
        stats[f] = {"mean": float(vals.mean()), "std": float(vals.std())}
    scaler = {
        "features": FEATURES, "window": WINDOW, "horizon_steps": HORIZON,
        "step_minutes": STEP_MIN, "feature_stats": stats,
        "target_temp_c": {"mean": float(train["target_temp_c"].mean()),
                          "std": float(train["target_temp_c"].std())},
    }
    (OUT_DIR / "scaler.json").write_text(json.dumps(scaler, indent=2))

    print(f"\nTrain windows: {len(train):,}   Test windows: {len(test):,}")
    print("Risk class share (train):")
    print(train["risk_class"].value_counts(normalize=True).sort_index().round(3))
    print(f"\nSaved files in {OUT_DIR}")


if __name__ == "__main__":
    main()

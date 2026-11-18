
from pathlib import Path
import re

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. PROJECT PATHS AND SETTINGS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "greenhouse_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "greenhouse_features_5min.csv"

# Our project uses a 5-minute sampling interval.
RESAMPLE_INTERVAL = "5min"


# --------------------------------------------------
# 2. POSSIBLE DATASET COLUMN NAMES
# --------------------------------------------------

COLUMN_ALIASES = {
    "timestamp": [
        "timestamp",
        "datetime",
        "date_time",
        "date",
        "time",
        "recorded_at"
    ],
    "temperature": [
        "temperature",
        "air_temperature",
        "air_temp",
        "temp",
        "tair",
        "temperature_c",
        "temperature_celsius"
    ],
    "humidity": [
        "humidity",
        "relative_humidity",
        "relativehumidity",
        "rh",
        "rh_percent",
        "humidity_percent"
    ]
}


def normalize_column_name(name):
    """Make column names easier to compare."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def find_column(columns, possible_names):
    """Find a dataset column using known aliases."""

    normalized_columns = {
        normalize_column_name(col): col
        for col in columns
    }

    for name in possible_names:
        key = normalize_column_name(name)

        if key in normalized_columns:
            return normalized_columns[key]

    return None


# --------------------------------------------------
# 3. LOAD AND IDENTIFY DATASET COLUMNS
# --------------------------------------------------

def load_dataset():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"\nDataset not found:\n{INPUT_FILE}\n"
            "Place your real greenhouse CSV inside data/raw/"
            " and name it greenhouse_data.csv."
        )

    df = pd.read_csv(INPUT_FILE)

    if df.empty:
        raise ValueError("The dataset is empty.")

    print("\nOriginal dataset columns:")
    print(df.columns.tolist())

    timestamp_col = find_column(
        df.columns, COLUMN_ALIASES["timestamp"]
    )

    temperature_col = find_column(
        df.columns, COLUMN_ALIASES["temperature"]
    )

    humidity_col = find_column(
        df.columns, COLUMN_ALIASES["humidity"]
    )

    missing = []

    if timestamp_col is None:
        missing.append("timestamp")

    if temperature_col is None:
        missing.append("temperature")

    if humidity_col is None:
        missing.append("humidity")

    if missing:
        raise ValueError(
            f"\nCould not identify these required columns: {missing}\n"
            "Check the dataset column names and update "
            "COLUMN_ALIASES in this script."
        )

    # Keep only the columns needed for this initial pipeline.
    df = df[
        [timestamp_col, temperature_col, humidity_col]
    ].copy()

    df.columns = [
        "timestamp",
        "temperature",
        "humidity"
    ]

    return df


# --------------------------------------------------
# 4. CLEAN DATA
# --------------------------------------------------

def clean_dataset(df):

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    df["temperature"] = pd.to_numeric(
        df["temperature"],
        errors="coerce"
    )

    df["humidity"] = pd.to_numeric(
        df["humidity"],
        errors="coerce"
    )

    # Remove invalid records.
    df = df.dropna(
        subset=["timestamp", "temperature", "humidity"]
    )

    # Remove physically invalid sensor values.
    df = df[
        (df["temperature"] > -30) &
        (df["temperature"] < 70) &
        (df["humidity"] > 0) &
        (df["humidity"] <= 100)
    ]

    df = df.sort_values("timestamp")

    # Average duplicate timestamps.
    df = (
        df.groupby("timestamp", as_index=False)
        [["temperature", "humidity"]]
        .mean()
    )

    if len(df) < 100:
        raise ValueError(
            "Too few valid records. Check the dataset."
        )

    return df


# --------------------------------------------------
# 5. CONVERT TO 5-MINUTE INTERVALS
# --------------------------------------------------

def resample_dataset(df):

    df = df.set_index("timestamp")

    # Average readings within each 5-minute interval.
    df = df.resample(RESAMPLE_INTERVAL).mean()

    # Do not invent missing sensor readings through interpolation.
    df = df.dropna(
        subset=["temperature", "humidity"]
    )

    return df.reset_index()


# --------------------------------------------------
# 6. CALCULATE DEW POINT, VPD AND DEW POINT DELTA
# --------------------------------------------------

def calculate_environmental_features(df):

    temperature = df["temperature"]
    humidity = df["humidity"]

    # Magnus-Tetens approximation.
    a = 17.27
    b = 237.7

    alpha = (
        (a * temperature) / (b + temperature)
        + np.log(humidity / 100.0)
    )

    dew_point = (b * alpha) / (a - alpha)

    # Saturation vapour pressure in kPa.
    saturation_vapour_pressure = (
        0.6108 * np.exp(
            (17.27 * temperature) / (temperature + 237.3)
        )
    )

    # Actual vapour pressure in kPa.
    actual_vapour_pressure = (
        saturation_vapour_pressure * humidity / 100.0
    )

    # Vapour Pressure Deficit in kPa.
    vpd = (
        saturation_vapour_pressure
        - actual_vapour_pressure
    )

    # Difference between air temperature and dew point.
    dew_point_delta = temperature - dew_point

    df["dew_point"] = dew_point
    df["vpd"] = vpd
    df["dew_point_delta"] = dew_point_delta

    return df


# --------------------------------------------------
# 7. SAVE PROCESSED DATASET
# --------------------------------------------------

def main():

    print("\n========== OMNINODE-AI DATA PIPELINE ==========")

    df = load_dataset()

    print(f"\nOriginal records: {len(df)}")

    df = clean_dataset(df)

    print(f"Valid records: {len(df)}")

    df = resample_dataset(df)

    print(f"Records after 5-minute resampling: {len(df)}")

    df = calculate_environmental_features(df)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nProcessed dataset saved successfully!")
    print(f"Output: {OUTPUT_FILE}")

    print("\nFinal columns:")
    print(df.columns.tolist())

    print("\nDataset preview:")
    print(df.head())

    print("\nDataset information:")
    print(df.describe())

    print("\n===============================================")


if __name__ == "__main__":
    main()

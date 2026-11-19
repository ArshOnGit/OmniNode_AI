"""temperature_forecasting.py - trains the 30-minute temperature forecaster."""
import json

import tensorflow as tf

from common import TRAINED, METADATA, load_scaler, load_split

SEED, EPOCHS, BATCH = 42, 100, 64


def main():
    tf.keras.utils.set_random_seed(SEED)
    scaler = load_scaler()
    X, y_temp, _ = load_split("training", scaler)
    t = scaler["target_temp_c"]
    y = (y_temp - t["mean"]) / t["std"]

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X.shape[1],)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    model.summary()

    early = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    model.fit(X, y, validation_split=0.1, epochs=EPOCHS, batch_size=BATCH,
              callbacks=[early], verbose=2)

    model.save(TRAINED / "temperature_model.keras")
    meta = {
        "task": "temperature_forecast",
        "window": scaler["window"],
        "features": scaler["features"],
        "horizon_steps": scaler["horizon_steps"],
        "step_minutes": scaler["step_minutes"],
        "input_length": int(X.shape[1]),
        "target_mean": t["mean"],
        "target_std": t["std"],
    }
    (METADATA / "temperature_metadata.json").write_text(json.dumps(meta, indent=2))
    print("Saved models/trained/temperature_model.keras")


if __name__ == "__main__":
    main()

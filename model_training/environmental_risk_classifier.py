"""environmental_risk_classifier.py - trains the Botrytis-risk classifier."""
import json

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from common import TRAINED, METADATA, load_scaler, load_split

SEED, EPOCHS, BATCH, NUM_CLASSES = 42, 100, 64, 3


def main():
    tf.keras.utils.set_random_seed(SEED)
    scaler = load_scaler()
    X, _, y = load_split("training", scaler)

    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X.shape[1],)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    early = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    model.fit(X, y, validation_split=0.1, epochs=EPOCHS, batch_size=BATCH,
              class_weight=class_weight, callbacks=[early], verbose=2)

    model.save(TRAINED / "risk_classifier.keras")
    meta = {
        "task": "risk_classification",
        "classes": ["low", "moderate", "high"],
        "window": scaler["window"],
        "features": scaler["features"],
        "input_length": int(X.shape[1]),
    }
    (METADATA / "classifier_metadata.json").write_text(json.dumps(meta, indent=2))
    print("Saved models/trained/risk_classifier.keras")


if __name__ == "__main__":
    main()

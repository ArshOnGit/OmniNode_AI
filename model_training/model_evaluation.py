"""model_evaluation.py - metrics and plots for both models on the test set."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, mean_absolute_error, mean_squared_error)

from common import RESULTS, RISK_NAMES, TRAINED, load_scaler, load_split


def evaluate_forecast(scaler):
    X, y_true, _ = load_split("test", scaler)
    model = tf.keras.models.load_model(TRAINED / "temperature_model.keras")
    t = scaler["target_temp_c"]
    pred = model.predict(X, verbose=0).ravel() * t["std"] + t["mean"]

    # baseline: "temperature stays the same as the latest reading"
    n_feat, win = len(scaler["features"]), scaler["window"]
    st = scaler["feature_stats"]["temp_c"]
    last_temp = X[:, (win - 1) * n_feat] * st["std"] + st["mean"]

    rows = []
    for name, p in [("FNN forecaster", pred), ("Persistence baseline", last_temp)]:
        rows.append({"model": name,
                     "MAE_C": mean_absolute_error(y_true, p),
                     "RMSE_C": float(np.sqrt(mean_squared_error(y_true, p)))})
    pd.DataFrame(rows).to_csv(RESULTS / "forecasting_metrics.csv", index=False)

    n = min(300, len(y_true))
    plt.figure(figsize=(10, 4))
    plt.plot(y_true[:n], label="Actual (+30 min)")
    plt.plot(pred[:n], label="Predicted", alpha=0.8)
    plt.xlabel("Test sample (5-min steps)")
    plt.ylabel("Temperature (C)")
    plt.title("30-minute temperature forecast")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "prediction_graph.png", dpi=200)
    plt.close()


def evaluate_classifier(scaler):
    X, _, y_true = load_split("test", scaler)
    model = tf.keras.models.load_model(TRAINED / "risk_classifier.keras")
    y_pred = np.argmax(model.predict(X, verbose=0), axis=1)

    report = classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=RISK_NAMES,
                                   output_dict=True, zero_division=0)
    df = pd.DataFrame(report).T
    df.loc["accuracy_overall", "precision"] = accuracy_score(y_true, y_pred)
    df.to_csv(RESULTS / "classification_metrics.csv")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    ConfusionMatrixDisplay(cm, display_labels=RISK_NAMES).plot(cmap="Blues", values_format="d")
    plt.title("Risk classifier - confusion matrix")
    plt.tight_layout()
    plt.savefig(RESULTS / "confusion_matrix.png", dpi=200)
    plt.close()


def main():
    scaler = load_scaler()
    evaluate_forecast(scaler)
    evaluate_classifier(scaler)
    print("Saved metrics and plots to results/")


if __name__ == "__main__":
    main()

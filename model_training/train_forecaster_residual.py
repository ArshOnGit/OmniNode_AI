#!/usr/bin/env python3
"""Train the improved OmniNode forecaster (residual / delta model).

Why: the old model predicted absolute T/RH at +30 min and lost to the
"persistence" baseline (predict the last reading). This model instead predicts
the CHANGE from the last reading and adds it back, and feeds the network the
last value plus differences to earlier samples. Differences are small numbers,
so they survive INT8 quantization far better than raw standardized values.

Run from the repository root (needs data/processed_greenhouse.csv):
    python model_training/train_forecaster_residual.py --seeds 5
Outputs go to model_training/artifacts_v2/<run-name>/.  Nothing in firmware/ is
modified; integrate by hand after reviewing report.md.
"""
import argparse, json, os, tempfile, time
from pathlib import Path
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/processed_greenhouse.csv")
ap.add_argument("--hidden", type=int, nargs="+", default=[16, 8])
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--epochs", type=int, default=200)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--name", default=None, help="run name (default: auto from settings)")
ap.add_argument("--out", default="model_training/artifacts_v2")
ap.add_argument("--smoke", action="store_true", help="tiny fast run to test the pipeline")
a = ap.parse_args()

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

W, STEP, H = 4, 5, 30  # must match the firmware window: 4 samples, 5 min apart, +30 min
df = pd.read_csv(a.data, parse_dates=["timestamp"])
if a.smoke:
    df = df.iloc[:6000]
ts = df.timestamp.values.astype("datetime64[m]").astype(np.int64)
if not np.all(np.diff(ts) > 0):
    raise SystemExit("timestamps must be strictly increasing")
V = df[["temperature_c", "relative_humidity_pct"]].to_numpy(np.float32)
n = len(df)
cut = [0, int(n * .70), int(n * .85), n]  # chronological 70/15/15 (same as the notebook)


def windows(lo, hi):
    pos = {t: i for i, t in enumerate(ts[lo:hi], lo)}
    X, y = [], []
    for i in range(lo + W - 1, hi):
        if not np.all(np.diff(ts[i - W + 1:i + 1]) == STEP):
            continue
        j = pos.get(ts[i] + H)
        if j is not None:
            X.append(V[i - W + 1:i + 1]); y.append(V[j])
    return np.array(X, np.float32), np.array(y, np.float32)


(Xtr, ytr), (Xva, yva), (Xte, yte) = [windows(cut[k], cut[k + 1]) for k in range(3)]
print(f"windows train/val/test: {len(Xtr)}/{len(Xva)}/{len(Xte)}")


def feats(X):
    last = X[:, -1, :]
    d = (X[:, :-1, :] - last[:, None, :]).reshape(len(X), -1)  # dT-15,dRH-15,dT-10,dRH-10,dT-5,dRH-5
    return np.concatenate([last, d], 1)


Ftr = feats(Xtr)
f_mean, f_std = Ftr.mean(0), Ftr.std(0) + 1e-6
Dtr = ytr - Xtr[:, -1, :]
t_mean, t_std = Dtr.mean(0), Dtr.std(0) + 1e-6
norm = lambda X: ((feats(X) - f_mean) / f_std).astype(np.float32)
tnorm = lambda X, y: ((y - X[:, -1, :] - t_mean) / t_std).astype(np.float32)
ntr, nva, nte = norm(Xtr), norm(Xva), norm(Xte)
ttr, tva = tnorm(Xtr, ytr), tnorm(Xva, yva)


def metrics(pred, true):
    e = pred - true
    return {"T_MAE": float(np.abs(e[:, 0]).mean()), "T_RMSE": float(np.sqrt((e[:, 0] ** 2).mean())),
            "RH_MAE": float(np.abs(e[:, 1]).mean()), "RH_RMSE": float(np.sqrt((e[:, 1] ** 2).mean()))}


persist = metrics(Xte[:, -1, :], yte)


def build(seed):
    tf.keras.utils.set_random_seed(seed)
    m = keras.Sequential([keras.Input(shape=(8,))] +
                         [layers.Dense(h, activation="relu") for h in a.hidden] + [layers.Dense(2)])
    m.compile(optimizer=keras.optimizers.Adam(a.lr), loss="mse")
    m.fit(ntr, ttr, validation_data=(nva, tva), epochs=2 if a.smoke else a.epochs, batch_size=a.batch,
          verbose=0, callbacks=[keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)])
    return m, float(m.evaluate(nva, tva, verbose=0))


def to_physical(raw_out, X):
    return X[:, -1, :] + raw_out * t_std + t_mean


runs = []
for seed in range(a.seeds):
    t0 = time.time()
    m, vloss = build(seed)
    mt = metrics(to_physical(m.predict(nte, verbose=0), Xte), yte)
    runs.append({"seed": seed, "val_loss": vloss, "float_test": mt, "model": m})
    print(f"seed {seed}: val_loss {vloss:.4f}  T MAE {mt['T_MAE']:.4f}  RH MAE {mt['RH_MAE']:.4f}  ({time.time()-t0:.0f}s)")

best = min(runs, key=lambda r: r["val_loss"])  # selection by VALIDATION only, never test
model = best["model"]

# ---- INT8 conversion -------------------------------------------------------
rep = ntr[np.linspace(0, len(ntr) - 1, min(256, len(ntr)), dtype=int)]


def convert_int8(converter):
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: ([r[None, :]] for r in rep)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


if callable(getattr(model, "export", None)):
    with tempfile.TemporaryDirectory() as saved_model_dir:
        model.export(saved_model_dir)
        conv = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir, signature_keys=["serve"])
        tfl = convert_int8(conv)
else:
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    tfl = convert_int8(conv)
it = tf.lite.Interpreter(model_content=tfl, experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
it.allocate_tensors()
i_d, o_d = it.get_input_details()[0], it.get_output_details()[0]
(i_s, i_z), (o_s, o_z) = i_d["quantization"], o_d["quantization"]
ops = sorted({d["op_name"] for d in it._get_ops_details()})


def int8_predict(Fn):
    out = np.empty((len(Fn), 2), np.float32)
    for k, row in enumerate(Fn):
        q = np.clip(np.round(row / i_s) + i_z, -128, 127).astype(np.int8)[None, :]
        it.set_tensor(i_d["index"], q); it.invoke()
        out[k] = (it.get_tensor(o_d["index"])[0].astype(np.float32) - o_z) * o_s
    return out


i8 = metrics(to_physical(int8_predict(nte), Xte), yte)
beats = all(i8[k] < persist[k] for k in persist)

# ---- artifacts ---------------------------------------------------------------
name = a.name or ("smoke" if a.smoke else "mlp_" + "x".join(map(str, a.hidden)))
out = Path(a.out) / name; out.mkdir(parents=True, exist_ok=True)
(out / "omninode_forecaster_v2_int8.tflite").write_bytes(tfl)
body = ",\n  ".join(", ".join(f"0x{b:02x}" for b in tfl[i:i + 12]) for i in range(0, len(tfl), 12))
(out / "omninode_forecaster_v2_int8.h").write_text(
    "#pragma once\n#include <cstddef>\n\nalignas(16) const unsigned char g_omninode_forecaster_v2_int8_model[] = {\n  "
    + body + "\n};\nconstexpr std::size_t g_omninode_forecaster_v2_int8_model_len = " + str(len(tfl)) + ";\n")
meta = {
    "model_version": 2, "input_window_samples": W, "sampling_interval_minutes": STEP, "forecast_horizon_minutes": H,
    "input_tensor_shape": [1, 8],
    "input_feature_order": ["T_last", "RH_last", "dT_t-15", "dRH_t-15", "dT_t-10", "dRH_t-10", "dT_t-5", "dRH_t-5"],
    "input_note": "d* = value at that time minus the LAST value (t). Standardize with input_mean/input_scale, then quantize.",
    "input_mean": f_mean.tolist(), "input_scale": f_std.tolist(),
    "input_quantization": {"scale": float(i_s), "zero_point": int(i_z)},
    "output_tensor_shape": [1, 2], "output_order": ["delta_temperature_c", "delta_relative_humidity_pct"],
    "output_note": "Dequantize, then delta = deq*output_scale + output_mean; forecast = last reading + delta.",
    "output_mean": t_mean.tolist(), "output_scale": t_std.tolist(),
    "output_quantization": {"scale": float(o_s), "zero_point": int(o_z)},
    "tflite_ops": ops, "tflite_bytes": len(tfl), "source_data": a.data, "hidden": a.hidden}
(out / "forecast_preprocessing_v2.json").write_text(json.dumps(meta, indent=2))
res = {"persistence_test": persist, "int8_test": i8, "int8_beats_persistence_on_all_4": beats,
       "selected_seed": best["seed"], "windows": [len(Xtr), len(Xva), len(Xte)],
       "per_seed_float_test": [{"seed": r["seed"], "val_loss": r["val_loss"], **r["float_test"]} for r in runs]}
(out / "results.json").write_text(json.dumps(res, indent=2))
rel = {k: f"{100*(1-i8[k]/persist[k]):+.1f}%" for k in persist}
(out / "report.md").write_text(
    f"# Run {name}\n\n| metric | persistence | INT8 model | improvement |\n|---|---:|---:|---:|\n" +
    "".join(f"| {k} | {persist[k]:.4f} | {i8[k]:.4f} | {rel[k]} |\n" for k in persist) +
    f"\nBeats persistence on all four metrics: **{beats}**. Selected seed {best['seed']} (by validation loss). "
    f"Ops: {ops}. Model size {len(tfl)} bytes.\n\nSeed spread (float, test): see results.json.\n")
print("\n" + (out / "report.md").read_text()); print("artifacts ->", out)

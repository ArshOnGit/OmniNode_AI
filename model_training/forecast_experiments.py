"""Compare forecast candidates against persistence on the chronological test
split (same windowing/split rules as feedfwdNN.ipynb). Run from repo root:
    python model_training/forecast_experiments.py
"""
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

W, STEP, H = 4, 5, 30
df = pd.read_csv("data/processed_greenhouse.csv", parse_dates=["timestamp"])
ts = df.timestamp.values.astype("datetime64[m]").astype(np.int64)
X_all = df[["temperature_c", "relative_humidity_pct"]].to_numpy(np.float32)
n = len(df)
cut = [0, int(n * .70), int(n * .85), n]

def windows(a, b):
    idx = {t: i for i, t in enumerate(ts[a:b], a)}
    X, y = [], []
    for i in range(a + W - 1, b):
        first = i - W + 1
        if not np.all(np.diff(ts[first:i + 1]) == STEP):
            continue
        j = idx.get(ts[i] + H)
        if j is None:
            continue
        X.append(X_all[first:i + 1]); y.append(X_all[j])
    return np.array(X), np.array(y)

parts = [windows(cut[k], cut[k + 1]) for k in range(3)]
(Xtr, ytr), (Xva, yva), (Xte, yte) = parts
print("windows train/val/test:", len(Xtr), len(Xva), len(Xte))

def report(name, pred):
    e = pred - yte
    mae = np.abs(e).mean(0); rmse = np.sqrt((e ** 2).mean(0))
    print(f"{name:34s} T MAE {mae[0]:.4f} RMSE {rmse[0]:.4f} | RH MAE {mae[1]:.4f} RMSE {rmse[1]:.4f}")

last = lambda X: X[:, -1, :]
report("persistence", last(Xte))

def feats(X):  # last value + differences relative to the last value
    l = last(X)
    return np.concatenate([l, (X[:, :-1, :] - l[:, None, :]).reshape(len(X), -1)], 1)

mu, sd = feats(Xtr).mean(0), feats(Xtr).std(0) + 1e-6
f = lambda X: (feats(X) - mu) / sd
dtr = ytr - last(Xtr)
for a in (1, 100):
    r = Ridge(alpha=a).fit(f(Xtr), dtr)
    report(f"ridge residual (alpha={a})", last(Xte) + r.predict(f(Xte)))
dsd = dtr.std(0)
for hid in ((8,), (16, 8)):
    m = MLPRegressor(hidden_layer_sizes=hid, early_stopping=True, max_iter=500, random_state=42).fit(f(Xtr), dtr / dsd)
    report(f"MLP residual {hid}", last(Xte) + m.predict(f(Xte)) * dsd)

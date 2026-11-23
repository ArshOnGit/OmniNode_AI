# OmniNode-AI: Progress Tracker

_Last updated: 2026-11-21. Source of truth for scope: `README.md` (and the custom-dashboard README variant, which replaces Blynk with a self-built dashboard)._

## Summary

| Area | Status |
|---|---|
| Firmware: DHT22 read, validation, dew point / VPD / margin, 60-sample history | Done (build-verified per README) |
| Firmware: on-device INT8 forecaster (+30 min T/RH) | Done (build-verified; not run on a board) |
| Firmware: provisional LOW/MODERATE/HIGH rule | Done (placeholder thresholds) |
| Data preprocessing (`preprocess_greenhouse.py`, 5-min windows, no interpolation) | Done; `data/processed_greenhouse.csv` exists (135,942 rows) |
| Forecast training notebook + embedded model | Done; **does not beat persistence baseline** |
| Serial telemetry (`@JSON` lines) for the dashboard | **New, this session** — not compiled (no PlatformIO in this env) |
| Dashboard bridge + web UI (`dashboard/`) | **New, this session** — tested with `--simulate`; not tested with a real board |
| Watanabe (2021) infection-risk index | **Done (code)**: `firmware/watanabe_risk.h` + `dashboard/watanabe.py`; C++ and Python agree in host tests; firmware not compiled on a board |
| Dashboard history (SQLite + CSV export) | **Done**, tested in simulation |
| Improved forecaster (residual model) | **Training script ready and smoke-tested; full training pending on the other laptop**; firmware not yet adapted to it |
| Trained ML risk classifier | Dropped for now: Watanabe index replaces it (no labels exist) |
| Hardware validation | Not started |

## Done

- ESP32 + DHT22 (GPIO 4) monitoring at 1/min; invalid readings skipped.
- Derived features: dew point (Magnus-Tetens), VPD, dew-point margin.
- TFLite Micro forecaster (4,304 B, 16 KiB arena, 5 ops). Build: 12.3% RAM, 24.2% flash.
- Forecast window logic: four valid samples 5 min apart; resets on a missed/invalid slot.
- Preprocessing pipeline with chronological 70/15/15 split, scaling fit on train only.

## Added this session

- `firmware/OmniNode_AI_Core.ino`: emits `@JSON {...}` lines (`sample` each minute, `forecast` every 5 min). Human-readable logs are unchanged; no Wi-Fi needed.
- `dashboard/bridge.py`: reads serial (needs `pip install pyserial`) or `--simulate`; serves `/api/state` and the UI on `localhost:8000`.
- `dashboard/index.html`: live cards, T/RH chart, forecast, risk. Shows a **SIMULATED** banner for fake data and flags stale data.

Run: `python dashboard/bridge.py --simulate` or `python dashboard/bridge.py --port /dev/ttyUSB0`.

## Leftover (priority order)

1. **Compile and flash** the updated firmware (`pio run -e esp32dev`); confirm `@JSON` lines parse in the bridge with a real DHT22.
2. **Hardware measurements:** inference latency, real arena use, heap/stack, sensor reliability, long-run stability.
3. **Dashboard transport decision** (see open questions): serial bridge now vs. Wi-Fi (HTTP/MQTT/WebSocket) later. Add history persistence if needed.
4. **Forecast model quality:** it loses to persistence (T MAE 0.575 vs 0.484; RH MAE 2.49 vs 1.85). Options: better features, a different model, or ship persistence as the baseline.
5. **Risk labeling protocol:** decide how to label (e.g., implement the Watanabe 2021 cumulative-infection-time method), reconcile the timezone-unknown timestamps, then build `environmental_risk_labeled.csv` and train the classifier.
6. **Sustained-humidity duration** feature (needed for any meaningful risk estimate).
7. **DHT22 vs. central-station domain gap:** collect real DHT22 time-series and re-evaluate.
8. **Repo hygiene:** `data/omninode_sensor_data.csv` is untimestamped and unusable; legacy headers `omninode_model.h` / `omninode_classifier.h` are unused; the custom-dashboard README has broken markdown escaping (`\*\*`, `\``) and should be merged into `README.md` once the dashboard direction is confirmed; many uncommitted changes in git.

## Open questions for discussion

1. Dashboard hosting: local laptop/Raspberry Pi over serial, or ESP32 on Wi-Fi pushing to a server? That decides the firmware work.
2. Should the dashboard store history (SQLite/CSV)? Doing so would also give the real DHT22 dataset in item 7.
3. Risk: keep the simple rule, or invest in the Watanabe cumulative method?
4. Forecast: improve the model, or present persistence as the honest baseline?
5. Any alerts needed (email/SMS/buzzer)?

## Session 2 update (decisions: USB serial, keep history, Watanabe 2021, improve model)

- Risk: Watanabe index implemented from the paper (coefficients read from the PDF). Our assumptions are listed at the top of `firmware/watanabe_risk.h` (Y clamped to 0 outside 3-30 C; the published polynomial gives Y(28 C)=0.26 vs the 0.50 measured ratio in the text; no RTC so the on-device index is a lower bound for 48 h after boot; the dashboard recomputes from stored history).
- Forecast: persistence reproduced exactly (T MAE 0.4837, RH MAE 1.8523). CPU-only experiments with a residual MLP (16x8) gave T MAE 0.4485 / RH MAE 1.8223, i.e. ~7% / ~1.6% better, on float models. Training moves to another laptop: see `TRAINING_GUIDE.md` and `../omninode_training_bundle.zip`.
- Next: (1) run training on the other laptop, send back the best `artifacts_v2/<run>/`; (2) adapt firmware to the 8-value input + delta output (resolver reduces to FULLY_CONNECTED only); (3) compile/flash; (4) collect real DHT22 history.

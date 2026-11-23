# OmniNode-AI: Tomato Greenhouse Environmental Monitoring

OmniNode-AI is an ESP32 and DHT22 research prototype for monitoring greenhouse air conditions, forecasting temperature and relative humidity, and showing provisional environmental conditions that may favor *Botrytis cinerea* (gray mold) in tomatoes.

**It does not detect fungal infection, diagnose disease, or control treatment.** An environmental-risk estimate is not evidence that plants are infected.

## Current status

- The firmware reads a DHT22 on GPIO 4 once per minute, validates readings, calculates dew point, vapor pressure deficit (VPD), and dew-point margin, and stores up to 60 one-minute samples.
- The firmware runs the 2,648-byte full-INT8 v2 residual temperature/RH forecaster locally. It uses four valid measurements five minutes apart (t−15, t−10, t−5, t), feeds the latest values and three historical differences to the model, and adds the predicted deltas to forecast both values at t+30. Forecasting starts after a 15-minute window is collected; missed or invalid five-minute readings clear the window.
- The firmware keeps the Watanabe 2021 48-hour cumulative index separate from the legacy point rule and from new current and +30-minute **demonstration environmental warnings**. None detects fungal infection or represents a future Watanabe score.
- Forecast initialization or inference errors are reported over Serial; sensor monitoring continues. The firmware does not require Wi-Fi. The current firmware has no Blynk code or credentials configured.
- The existing `data/omninode_sensor_data.csv` has only `temperature,humidity` columns and no timestamps. It is not suitable for temporal forecasting or scientifically grounded risk labeling.
- The commercial tomato greenhouse dataset contains central-station measurements, not Botrytis infection or risk labels. The firmware model was trained and evaluated with those data; its results are not validation of DHT22 deployment performance.
- `firmware/omninode_forecaster_v2_int8.h` contains the exact bytes from the selected `mlp_16x8` model. Keep this header paired with that run's `forecast_preprocessing_v2.json`; the legacy model headers are not used by this sketch.

## Hardware and build

- ESP32 NodeMCU (`esp32dev` PlatformIO target)
- DHT22 data connected to GPIO 4
- PlatformIO with the Arduino framework

Install PlatformIO, connect the board, then run these commands from the repository root:

```text
pio run -e esp32dev
pio run -e esp32dev -t upload
pio device monitor -b 115200
```

Dependencies and versions are declared in [`platformio.ini`](platformio.ini). TensorFlow Lite Micro 1.26.46 runs the forecast with a single 16 KiB tensor arena and a resolver containing only the model's required `FULLY_CONNECTED` operator. The project and library build use C++17.

The integrated PlatformIO build succeeded for `esp32dev`. Its linker summary reported **45,804 bytes of static RAM out of 327,680 bytes (14.0%)** and **309,636 bytes of flash out of 1,310,720 bytes (23.6%)**. Static RAM includes the reserved 16 KiB tensor arena; it is not a measurement of peak runtime heap/stack use. On device, the sketch prints actual TFLite Micro arena use after allocation and inference time after each forecast. Neither has been measured on physical hardware yet.

## Environmental features

For each valid DHT22 reading, the firmware calculates:

- Dew point with the Magnus-Tetens approximation
- VPD in kPa as saturation vapor pressure minus actual vapor pressure
- Dew-point margin as air temperature minus dew point

The 60-sample history is for monitoring. Separately, the forecast uses only four valid samples from five-minute slots. If a scheduled five-minute reading is invalid or a slot is missed, the forecast window is restarted rather than filling missing values.

## Dataset preparation

The selected source is the [Mendeley Data commercial tomato greenhouse dataset](https://doi.org/10.17632/tkbkzdt5nr.1). For forecasting, the current workflow uses the 2018 and 2019 `CENTRAL.xlsx` measurements: temperature in degrees Celsius and relative humidity in percent, recorded every five minutes. These central-station readings are not Botrytis labels, and their sensor setup differs from the deployment DHT22.

The Excel timestamps do not specify a timezone. The preprocessing pipeline preserves them as naive timestamps and marks the timezone `unknown`; it does not convert them to UTC or assume a local timezone. The 2019 central series contains a long gap. Missing measurements are dropped, and gaps remain unfilled.

Run the preprocessor on the two central-station workbooks, substituting the path where you keep the dataset:

```text
python model_training\preprocess_greenhouse.py "C:\path\to\dataset\T&RH\2018\CENTRAL.xlsx" "C:\path\to\dataset\T&RH\2019\CENTRAL.xlsx" --output data\processed_greenhouse.csv --timestamp-column Timestep --temperature-column Temperature --humidity-column RelativeHumidity --temperature-unit C --humidity-unit percent --timestamp-timezone unknown --sample-interval-minutes 5
```

The script also accepts compatible CSV inputs. `--resample-one-minute` is not appropriate for this five-minute source: slower data must not be upsampled or interpolated. The script calculates environmental features and creates a 15-minute window only from four actual readings at five-minute intervals. It creates a forecast target only when the exact +30-minute timestamp is present.

Keep the raw dataset outside the repository. Do not commit large raw files; use Git LFS or document the dataset source and access steps instead. This project does not download or fabricate datasets.

For a Google Colab runtime, after opening or cloning the repository and setting the working directory to its root, install the notebook dependencies before running training cells:

```text
%pip install -r model_training/requirements.txt
```

Training uses Python 3.10–3.12. TensorFlow has no wheels for newer Python releases on many platforms. On a training laptop, create a virtual environment and install the training requirements:

```text
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r model_training\requirements.txt
```

The dashboard has no dependency for simulation mode:

```text
python dashboard\bridge.py --simulate
```

To display a saved Notepad serial log without connecting to the ESP32, save the
log as a UTF-8 text file. In PowerShell, from the `OmniNode_AI` project root,
run:

```text
$log = Read-Host "Full path to the saved Notepad log"
$db = Join-Path $env:TEMP ("omninode-replay-{0}.sqlite" -f (Get-Date -Format "yyyyMMdd-HHmmss"))
python .\dashboard\bridge.py --replay "$log" --db "$db"
```

Replay reads `@JSON` sample and forecast records, ignores ordinary Serial log
lines, and preserves sample spacing from `uptime_s` when calculating the
dashboard's Watanabe history coverage. It uses a separate `replay` history
source and does not open a COM port. Open `http://localhost:8000`; leave the
PowerShell process running while viewing the dashboard. Use a new `--db` path
for a fresh replay history.

For generated or otherwise synthetic replay data, add `--demo-data` so the
dashboard explicitly labels the display and matched comparisons as demo-only:

```text
python .\dashboard\bridge.py --replay "$log" --db "$db" --demo-data
```

The dashboard frontend provides Overview, Monitoring, AI Forecast, Risk
Analysis, Analytics, and Device Logs views. It uses the bridge's existing
`/api/state` and `/api/export.csv` endpoints; forecast error metrics are
calculated only from matched future samples, and replay is labeled offline.
Watanabe coverage remains provisional until 48 hours are available.

Only a real board connected over USB needs `pyserial`. Install it with:

```text
python -m pip install -r dashboard\requirements.txt
python dashboard\bridge.py --port COM3
```

## Forecasting workflow

`model_training/train_forecaster_residual.py` trains the selected residual v2 model from `data/processed_greenhouse.csv`. It uses four actual readings at t−15, t−10, t−5, and t; chronological 70/15/15 train, validation, and test partitions; and no windows or targets that cross a partition boundary. The ESP32 samples once per minute and runs inference every five minutes after four valid five-minute readings have filled the window.

The model input tensor is `[1, 8]`, ordered as the latest temperature and humidity followed by the temperature and humidity differences (historical reading minus latest) at t−15, t−10, and t−5. Firmware normalization and quantization values are stored with the selected model in `forecast_preprocessing_v2.json`: input quantization scale `0.05126921087503433`, zero point `-17`; output quantization scale `0.011703990399837494`, zero point `1`. The `[1, 2]` output represents temperature and humidity deltas; after dequantization and output inverse-scaling, firmware adds each delta to the latest corresponding reading. Keep the model header and metadata from the same run paired.

The selected `mlp_16x8` model was evaluated on 20,365 held-out central-station windows and beat persistence on all four metrics:

| Target | INT8 MAE | INT8 RMSE | Persistence MAE | Persistence RMSE |
|---|---:|---:|---:|---:|
| Temperature (°C) | 0.4465 | 0.6284 | 0.4837 | 0.6760 |
| Relative humidity (%) | 1.8055 | 2.6056 | 1.8523 | 2.7833 |

These are dataset results, not ESP32/DHT22 deployment measurements or evidence of disease prediction accuracy.

## Environmental-risk workflow

`model_training/Environmental_Classifier.ipynb` estimates three **environmental** categories:

| Class | Meaning |
|---|---|
| 0 | Low environmental risk |
| 1 | Moderate environmental risk |
| 2 | High environmental risk |

The base features are temperature, relative humidity, VPD, and dew-point margin. Sustained high-humidity duration is optional and may only be used when its calculation and provenance are documented.

The notebook requires a timestamped `data/environmental_risk_labeled.csv` containing `timestamp_utc`, the feature columns, `environmental_risk_label`, and `rule_based_risk_label`. The selected microclimate dataset has no Botrytis labels, and its timezone-unknown forecast timestamps must not be converted simply to meet this classifier input schema. Reconcile timestamp handling and establish an approved, documented environmental-risk labeling protocol before preparing classifier data. No trained classifier is deployed by the firmware.

The firmware reports three distinct environmental indicators:

1. **Watanabe 2021 cumulative index:** the existing rolling 48-hour effective-infection-time calculation. Its score and classification are unchanged. Coverage is reported as warming up until 48 hours of valid history are available.
2. **Current environmental warning:** a snapshot rule applied to the live DHT22 temperature, RH, VPD, and dew-point margin.
3. **Predicted environmental warning:** the same snapshot rule applied to forecast temperature and RH after the firmware recomputes predicted dew point, VPD, and dew-point margin. This is not a future Watanabe score.

The new warning cutoffs are explicitly **configurable demonstration thresholds**, not scientifically validated infection-risk thresholds. HIGH is emitted when all conditions match: RH ≥ 90%, temperature 15–25 °C, VPD ≤ 0.45 kPa, and dew-point margin ≤ 2.1 °C. Otherwise, CAUTION is emitted when all conditions match: RH ≥ 85%, temperature 10–30 °C, VPD ≤ 0.80 kPa, and dew-point margin ≤ 3.5 °C. All other conditions are LOW. The values are centralized in [`environmental_warning.h`](firmware/environmental_warning.h). VPD and dew-point margin are derived from the same air temperature and RH; they are not independent sensors, and the device does not measure leaf wetness.

The existing `simple_rule_risk` remains separate and unchanged: HIGH at RH ≥ 90% and 15–25 °C; otherwise MODERATE at RH ≥ 85% and 10–30 °C; LOW otherwise. Watanabe, H., Kojima, K., Kudomi, S., and Shimazu, T. (2021), “Estimation of greenhouse tomato infection risk by *Botrytis cinerea* based on temperature and relative humidity,” *Annual Report of the Kansai Plant Protection Society*, 63, 59–65, uses a two-day cumulative effective-infection-time approach, a temperature-dependent fungal-growth adjustment, and a dew-point-margin condition. [DOI: 10.4165/kapps.63.59](https://doi.org/10.4165/kapps.63.59). That method does not validate the new instantaneous warning cutoffs. None of these environmental indicators confirms infection or diagnoses disease.

## Limitations and evaluation still needed

- DHT22-only measurements describe ambient air and do not directly measure leaf wetness, plant condition, or pathogen presence.
- Environmental favorability is not confirmed infection, disease diagnosis, or treatment advice.
- The new instantaneous warning thresholds are demonstration settings and are not calibrated for this greenhouse; the Watanabe assessment remains a lower bound while its 48-hour history warms up.
- The residual v2 forecast beats persistence on its central-station test windows, but DHT22 deployment validation remains future work.
- Physical-board inference latency, actual arena use, runtime heap/stack behavior, sensor reliability, and offline operation have not yet been hardware-tested. Build-time static RAM/flash figures above are not board measurements.
- Validate the data protocol, labeling criteria, split methodology, model metrics, and on-device behavior before making research or deployment claims.

## Repository layout

```text
OmniNode_AI/
├── README.md
├── platformio.ini
├── data/
│   └── omninode_sensor_data.csv
├── firmware/
│   ├── OmniNode_AI_Core.ino
│   ├── environmental_warning.h # Configurable demo warning rules and shared metrics
│   ├── omninode_forecaster_v2_int8.h # Embedded 2,648-byte INT8 residual model
│   ├── omninode_model.h              # Legacy header; not used by current firmware
│   └── omninode_classifier.h         # Legacy header; not used by current firmware
└── model_training/
    ├── preprocess_greenhouse.py
    ├── train_forecaster_residual.py
    ├── feedfwdNN.ipynb
    └── Environmental_Classifier.ipynb
```

# Training the improved forecaster on another laptop

Training is a small job (an 8-input, ~200-parameter network on ~95k windows): minutes on any laptop, CPU only, no GPU. Nothing here needs to run on the development laptop.

## 1. What to copy to the training laptop
Use `omninode_training_bundle.zip` (made next to this repo), or copy by hand:
- `model_training/train_forecaster_residual.py`
- `model_training/requirements.txt`
- `data/processed_greenhouse.csv` (18 MB — **git-ignored, so it will NOT arrive via `git clone`**; copy it by USB/cloud, or regenerate it, see section 4)

Keep the folder layout (`model_training/…`, `data/…`) and run everything from the folder that contains them.

## 2. Setup (Python 3.10–3.12)
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r model_training/requirements.txt
```
Note: the TensorFlow wheels need Python ≤ 3.12 on most platforms. Colab also works (`%pip install -r model_training/requirements.txt`, upload the files).

## 3. Run
```bash
python model_training/train_forecaster_residual.py --smoke --seeds 1     # 1-minute pipeline check
python model_training/train_forecaster_residual.py --seeds 5             # real run, default 16x8
```
Iterations to try (each writes its own folder under `model_training/artifacts_v2/<name>/`):
```bash
python model_training/train_forecaster_residual.py --seeds 5 --hidden 16 8   --name mlp_16x8
python model_training/train_forecaster_residual.py --seeds 5 --hidden 32 16  --name mlp_32x16
python model_training/train_forecaster_residual.py --seeds 5 --hidden 8      --name mlp_8
python model_training/train_forecaster_residual.py --seeds 5 --hidden 16 8 --lr 3e-4 --name mlp_16x8_lr3e-4
```
Each run prints per-seed results and writes `report.md`. **Pick the run by `report.md` where the INT8 model beats persistence on all four metrics**; the seed inside a run is chosen on validation loss, never test.

## 4. Where the training data comes from
- Source: Mendeley Data, commercial tomato greenhouse dataset, <https://doi.org/10.17632/tkbkzdt5nr.1> (free download; this project never downloads or fabricates data).
- Only `T&RH/2018/CENTRAL.xlsx` and `T&RH/2019/CENTRAL.xlsx` are used (5-minute temperature/RH).
- `data/processed_greenhouse.csv` is already produced from them (135,942 rows). Regenerate only if needed:
  `python model_training/preprocess_greenhouse.py <2018 CENTRAL.xlsx> <2019 CENTRAL.xlsx> --output data/processed_greenhouse.csv --timestamp-column Timestep --temperature-column Temperature --humidity-column RelativeHumidity --temperature-unit C --humidity-unit percent --timestamp-timezone unknown --sample-interval-minutes 5` (also needs `pip install openpyxl`).
- **Your own DHT22 data** (the data that matters for deployment): the dashboard stores every reading in `dashboard/history.sqlite`; `http://localhost:8000/api/export.csv` downloads it. A few weeks of it is needed before it can validate or fine-tune anything.

## 5. What to send back
The whole folder `model_training/artifacts_v2/<best run>/` (5 small files: `.tflite`, `.h`, `forecast_preprocessing_v2.json`, `results.json`, `report.md`). The selected `mlp_16x8` model is already integrated in the current firmware. For a different run, do not copy its header alone into `firmware/`; the matching preprocessing metadata and model-specific firmware changes must be integrated together.

## What the numbers mean
Persistence (repeat the last reading) is a strong baseline for 30-minute forecasts. Expect modest gains (a few percent; CPU experiments on this data gave about 2–7%). Gains on central-station data are not proof of DHT22 performance.

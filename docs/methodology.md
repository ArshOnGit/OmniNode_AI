# Methodology
1. Preprocessing (data_preprocessing.py)
2. Features: dew point, DPD, VPD
3. Risk labels: rule-based (DPD <= 1 C or RH >= 90% = high; DPD <= 2.5 C or RH >= 85% or VPD < 0.4 kPa = moderate; else low). Indicators of favourable conditions, not proof of infection.
4. Models: FNN 75-32-16-1 (forecast), 75-32-16-3 (risk), Adam, early stopping, balanced class weights
5. Evaluation: MAE/RMSE vs persistence baseline; accuracy, per-class precision/recall, confusion matrix
6. Deployment: int8 post-training quantization, TFLite Micro on ESP32

# System architecture
Sense: DHT22 -> ESP32 (reading every 30 s, averaged to 5-minute samples)
Analyse: dew point, dew point delta, VPD (Magnus-Tetens) on the ESP32
Predict: int8 FNN forecasts temperature 30 min ahead; int8 FNN classifies Botrytis-favourable risk (low/moderate/high)
Act: buzzer and relay run locally; Blynk dashboard when Wi-Fi is available

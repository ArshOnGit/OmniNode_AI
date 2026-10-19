# OmniNode-AI: Edge-Intelligent Multipurpose IoT Node

An advanced, low-cost, multipurpose IoT monitoring node powered by an **ESP32** and **DHT22** sensor, designed for smart environments (greenhouses, industrial facilities, and smart spaces).

## Key Features
- **Local Edge Analytics:** Computes high-precision **Dew Point** (via Magnus-Tetens formula) and **Heat Index** (via NOAA regression equation) directly on the microcontroller.
- **Edge AI Integration:** Deploys lightweight TinyML models for *Thermal Spike Forecasting* and *Environmental State Recognition*.
- **Cloud & Mobile Dashboard:** Streams real-time telemetry and historical time-series data to a mobile interface via Blynk IoT.

## Tech Stack
- **Hardware:** ESP32 NodeMCU, DHT22 Sensor
- **Firmware:** Arduino IDE (C++)
- **Edge AI:** TensorFlow Lite for Microcontrollers / Edge Impulse
- **Dashboard:** Blynk IoT Cloud

## Project Team
- **Arsh** ([https://github.com/ArshOnGit])
- **Preetish Singh** ([https://github.com/PreetishSingh])
- **Suraj Durgapal** ([https://github.com/Suraj-Durgapal])

## Repository Structure

```text
OmniNode-AI/
├── README.md                            # Comprehensive project docs, setup instructions, and architecture overview
│
├── firmware/
│   ├── OmniNode_AI_Core.ino             # Main ESP32 firmware: DHT22 telemetry, Magnus-Tetens dew point,
│   │                                    # NOAA heat index, circular windowing, & dual TFLite model inference
│   ├── omninode_model.h                 # Exported C++ header for temperature forecasting weights
│   └── omninode_classifier.h            # Exported C++ header for environmental state classification weights
│
├── model_training/
│   ├── feedfwdNN.ipynb                  # Colab notebook: Forecasting model training, validation, & export
│   └── environmental_classifier.ipynb   # Colab notebook: State classifier training, evaluation, & export
│
├── data/
    └── omninode_sensor_data.csv         # Synthetic time-series telemetry dataset used for model training


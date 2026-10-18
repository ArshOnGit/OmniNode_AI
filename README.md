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
│
├── firmware/
│   ├── OmniNode_AI_Core.ino   # Main ESP32 firmware: handles DHT22 reads, Magnus-Tetens dew point,
│   │                          # NOAA heat index, circular windowing, and live TFLite inference.
│   └── omninode_model.h       # Exported C++ header containing the quantized neural network weights.
│
├── model_training/
│   └── feedfwdNN.ipynb        # Google Colab notebook for data generation, training, validation (for temprature prediction)
│                              # error metrics (MAE/RMSE), and TFLite model conversion.
│
└── data/
    └── omninode_sensor_data.csv # Synthetic time-series telemetry dataset used for model training in feedfwdNN.ipynb 

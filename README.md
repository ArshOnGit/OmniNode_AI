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
- 

## Prerequisites & Arduino IDE Setup

If you are a collaborator working with this firmware, you will need to install the following dependencies inside your **Arduino IDE** before compiling `OmniNode_AI_Core.ino`:

### 1. ESP32 Board Support
Ensure your Arduino IDE has official support for Espressif ESP32 boards:
- Go to **File > Preferences**
- Add this URL to *Additional Boards Manager URLs*: 
  `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
- Go to **Tools > Board > Boards Manager**, search for **ESP32**, and install version `2.x` or higher.

### 2. Required Arduino Libraries
Open the **Library Manager** (`Ctrl+Shift+P` or **Tools > Manage Libraries**), search for, and install the following libraries:
* **`DHT sensor library`** (by Adafruit) — *Handles interface reads for the DHT22.*
* **`Adafruit Unified Sensor`** (by Adafruit) — *Required dependency for the DHT library.*
* **`TensorFlowLite_ESP32`** (by QuantumLeap / TensorFlow authors) — *Enables local TFLite micro-interpreter execution on the ESP32 core.*


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

```

## What's Done (Completed Architecture)
- **Sensor Telemetry & Math**: 
  - Integrated Magnus-Tetens formula for exact dew point calculation.
  - Integrated NOAA Regression equation for standard heat index calculation.
- **TinyML Forecasting**: 
  - Trained a sliding-window FNN model in Google Colab (MAE ~0.83°C).
  - Exported and quantized model weights into `omninode_model.h`.
- **Environmental State Classification**: 
  - Trained a multi-class neural network classifier (90% accuracy) to categorize operational states: *Stable/Idle*, *Warming/Active*, and *Cooling/Ventilation*.
  - Exported weights into `omninode_classifier.h`.
- **Firmware Logic (`OmniNode_AI_Core.ino`)**: 
  - Combines sensor polling, mathematical equations, dual TFLite model interpreters, and a 60-point circular RAM rolling buffer.

---

## What's Remaining (Future Deployment)
- **Physical Hardware Wiring**: Wiring the physical ESP32 development board to a DHT22 sensor module.
- **Wi-Fi & Blynk App Integration**: Injecting live Wi-Fi credentials (`ssid`/`pass`), adding the Blynk IoT library connection, and mapping the RAM history arrays (`blynkTempHistory`) to Blynk Virtual Pins and SuperChart.


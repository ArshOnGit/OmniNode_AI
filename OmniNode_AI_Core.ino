/**
 * @file OmniNode_Sensor_Engine.ino
 * @brief Multipurpose IoT Monitoring Node (OmniNode-AI) with TinyML Forecasting
 * @details Reads DHT22, calculates Magnus-Tetens Dew Point, NOAA Heat Index, 
 *          and runs a TinyML neural network model locally on the ESP32 for temperature prediction.
 * @author Arsh
 * @date October 2026
 */

#include <DHT.h>
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "omninode_model.h" // Your trained neural network model header

#define DHTPIN 4      
#define DHTTYPE DHT22   

DHT dht(DHTPIN, DHTTYPE);

// TensorFlow Lite Globals for ESP32
const int kArenaSize = 2048;
uint8_t tensor_arena[kArenaSize];
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Sliding window buffer to store past 15 temperature readings
const int windowSize = 15;
float tempBuffer[windowSize];
int bufferIndex = 0;
bool bufferFilled = false;

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n[INFO] OmniNode-AI: Initializing Sensor & TinyML Engine...");
  dht.begin();

  // Initialize TFLite Model
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      tflite::GetModel(omninode_model), resolver, tensor_arena, kArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("[ERROR] Tensor allocation failed!");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);
  Serial.println("[INFO] TinyML Model Loaded Successfully!");
}

void loop() {
  delay(2000); // Sample every 2 seconds

  float humidity = dht.readHumidity();
  float tempC = dht.readTemperature();

  if (isnan(humidity) || isnan(tempC)) {
    Serial.println("[ERROR] Failed to read from DHT sensor!");
    return;
  }

  // 1. Magnus-Tetens Dew Point Calculation
  float a = 17.27;
  float b = 237.7;
  float alpha = ((a * tempC) / (b + tempC)) + log(humidity / 100.0);
  float dewPoint = (b * alpha) / (a - alpha);

  // 2. NOAA Regression Heat Index Calculation
  float tempF = (tempC * 9.0 / 5.0) + 32.0;
  float heatIndexF = -42.379 + 2.04901523 * tempF + 10.14333127 * humidity
                   - 0.22475541 * tempF * humidity - 0.00683783 * tempF * tempF
                   - 0.05481717 * humidity * humidity + 0.00122874 * tempF * tempF * humidity
                   + 0.00085282 * tempF * humidity * humidity - 0.00000199 * tempF * tempF * humidity * humidity;
  float heatIndexC = (heatIndexF - 32.0) * 5.0 / 9.0;

  // 3. Update Sliding Window & Run TinyML Prediction
  tempBuffer[bufferIndex] = tempC;
  bufferIndex = (bufferIndex + 1) % windowSize;
  if (bufferIndex == 0) bufferFilled = true;

  // Serial Telemetry Output
  Serial.println("----------------------------------------");
  Serial.print("Ambient Temperature: "); Serial.print(tempC); Serial.println(" °C");
  Serial.print("Relative Humidity:   "); Serial.print(humidity); Serial.println(" %");
  Serial.print("Dew Point (Magnus):  "); Serial.print(dewPoint); Serial.println(" °C");
  Serial.print("Heat Index (NOAA):   "); Serial.print(heatIndexC); Serial.println(" °C");

  // Only run AI forecast once we have a full window of 15 samples
  if (bufferFilled) {
    // Feed the sliding window into the neural network model input tensor
    for (int i = 0; i < windowSize; i++) {
      int idx = (bufferIndex + i) % windowSize; // Reorder chronologically
      input->data.f[i] = tempBuffer[idx];
    }

    // Run inference (prediction)
    if (interpreter->Invoke() == kTfLiteOk) {
      float predictedTemp = output->data.f[0];
      Serial.print("AI Predicted Temp:   "); Serial.print(predictedTemp); Serial.println(" °C (30-min outlook)");
    } else {
      Serial.println("[ERROR] Model inference failed.");
    }
  } else {
    Serial.print("AI Model Warming Up... (");
    Serial.print(bufferIndex);
    Serial.println("/15 samples collected)");
  }
}

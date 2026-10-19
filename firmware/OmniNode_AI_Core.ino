/**
 * @file OmniNode_AI_Core.ino
 * @brief Multipurpose IoT Monitoring Node (OmniNode-AI) with Dual TinyML Models & RAM Rolling Buffer
 * @details Reads DHT22, calculates Magnus-Tetens Dew Point, NOAA Heat Index,
 *          runs FNN Forecasting, State Classification, and manages a RAM circular buffer for Blynk.
 * @author Arsh
 * @date October 2026
 */

#include <DHT.h>
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "omninode_model.h"          // Forecasting model header
#include "omninode_classifier.h"     // Environmental state classifier header

#define DHTPIN 4      
#define DHTTYPE DHT22   

DHT dht(DHTPIN, DHTTYPE);

// TensorFlow Lite Globals for ESP32
const int kArenaSize = 4096;
uint8_t tensor_arena[kArenaSize];

tflite::MicroInterpreter* forecaster_interpreter = nullptr;
TfLiteTensor* forecaster_input = nullptr;
TfLiteTensor* forecaster_output = nullptr;

tflite::MicroInterpreter* classifier_interpreter = nullptr;
TfLiteTensor* classifier_input = nullptr;
TfLiteTensor* classifier_output = nullptr;

// 1. Sliding window buffer for TinyML models (15 samples)
const int windowSize = 15;
float tempBuffer[windowSize];
int bufferIndex = 0;
bool bufferFilled = false;

// 2. RAM Rolling Buffer (Circular Buffer) for Blynk SuperChart History (e.g., last 60 points)
const int historyCapacity = 60;
float blynkTempHistory[historyCapacity];
float blynkHumidityHistory[historyCapacity];
int historyHead = 0;
int historyCount = 0;

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n[INFO] OmniNode-AI: Initializing Sensor, Dual Models & RAM Rolling Buffer...");
  dht.begin();

  static tflite::AllOpsResolver resolver;

  // Initialize Temperature Forecasting Model
  static tflite::MicroInterpreter static_forecaster(
      tflite::GetModel(omninode_model), resolver, tensor_arena, kArenaSize);
  forecaster_interpreter = &static_forecaster;
  if (forecaster_interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("[ERROR] Forecasting model tensor allocation failed!");
    return;
  }
  forecaster_input = forecaster_interpreter->input(0);
  forecaster_output = forecaster_interpreter->output(0);

  // Initialize Environmental State Classifier Model
  static tflite::MicroInterpreter static_classifier(
      tflite::GetModel(omninode_classifier), resolver, tensor_arena + 2048, 2048);
  classifier_interpreter = &static_classifier;
  if (classifier_interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("[ERROR] Classifier model tensor allocation failed!");
    return;
  }
  classifier_input = classifier_interpreter->input(0);
  classifier_output = classifier_interpreter->output(0);

  Serial.println("[INFO] System Ready & Fully Initialized!");
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

  // 3. Update TinyML Sliding Window
  tempBuffer[bufferIndex] = tempC;
  bufferIndex = (bufferIndex + 1) % windowSize;
  if (bufferIndex == 0) bufferFilled = true;

  // 4. Update RAM Rolling Buffer for Blynk SuperChart History
  blynkTempHistory[historyHead] = tempC;
  blynkHumidityHistory[historyHead] = humidity;
  historyHead = (historyHead + 1) % historyCapacity;
  if (historyCount < historyCapacity) historyCount++;

  // Serial Telemetry Output
  Serial.println("----------------------------------------");
  Serial.print("Ambient Temperature: "); Serial.print(tempC); Serial.println(" °C");
  Serial.print("Relative Humidity:   "); Serial.print(humidity); Serial.println(" %");
  Serial.print("Dew Point (Magnus):  "); Serial.print(dewPoint); Serial.println(" °C");
  Serial.print("Heat Index (NOAA):   "); Serial.print(heatIndexC); Serial.println(" °C");
  Serial.print("RAM History Buffer:  "); Serial.print(historyCount); Serial.println("/60 points logged");

  // Run AI inference once we have a full window of 15 samples
  if (bufferFilled) {
    float orderedWindow[windowSize];
    for (int i = 0; i < windowSize; i++) {
      int idx = (bufferIndex + i) % windowSize;
      orderedWindow[i] = tempBuffer[idx];
    }

    // --- A. Run Temperature Forecasting ---
    for (int i = 0; i < windowSize; i++) {
      forecaster_input->data.f[i] = orderedWindow[i];
    }
    if (forecaster_interpreter->Invoke() == kTfLiteOk) {
      float predictedTemp = forecaster_output->data.f[0];
      Serial.print("AI Forecast Temp:    "); Serial.print(predictedTemp); Serial.println(" °C (30-min outlook)");
    }

    // --- B. Run Environmental State Classification ---
    for (int i = 0; i < windowSize; i++) {
      classifier_input->data.f[i] = orderedWindow[i];
    }
    if (classifier_interpreter->Invoke() == kTfLiteOk) {
      float probIdle = classifier_output->data.f[0];
      float probWarming = classifier_output->data.f[1];
      float probCooling = classifier_output->data.f[2];

      int predictedState = 0;
      float maxProb = probIdle;
      if (probWarming > maxProb) { maxProb = probWarming; predictedState = 1; }
      if (probCooling > maxProb) { maxProb = probCooling; predictedState = 2; }

      Serial.print("AI State Class:      ");
      if (predictedState == 0) Serial.println("Stable / Idle");
      else if (predictedState == 1) Serial.println("Warming / Active");
      else Serial.println("Cooling / Ventilation");
    }

  } else {
    Serial.print("AI Models Warming Up... (");
    Serial.print(bufferIndex);
    Serial.println("/15 samples collected)");
  }
}

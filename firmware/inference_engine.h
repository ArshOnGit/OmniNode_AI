#pragma once
#include <Arduino.h>
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "model_config.h"
#include "model_headers/temperature_model.h"
#include "model_headers/risk_classifier.h"

constexpr int ARENA_BYTES = 10 * 1024;   // increase if AllocateTensors fails
alignas(16) static uint8_t tempArena[ARENA_BYTES];
alignas(16) static uint8_t riskArena[ARENA_BYTES];

static tflite::AllOpsResolver opsResolver;
static tflite::MicroInterpreter* tempInterp = nullptr;
static tflite::MicroInterpreter* riskInterp = nullptr;
static TfLiteTensor* tempIn = nullptr;
static TfLiteTensor* tempOut = nullptr;
static TfLiteTensor* riskIn = nullptr;
static TfLiteTensor* riskOut = nullptr;

inline bool initInference() {
  static tflite::MicroErrorReporter errReporter;
  const tflite::Model* tm = tflite::GetModel(temperature_model);
  const tflite::Model* rm = tflite::GetModel(risk_classifier);
  if (tm->version() != TFLITE_SCHEMA_VERSION || rm->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch");
    return false;
  }
  static tflite::MicroInterpreter tInterp(tm, opsResolver, tempArena, ARENA_BYTES, &errReporter);
  static tflite::MicroInterpreter rInterp(rm, opsResolver, riskArena, ARENA_BYTES, &errReporter);
  if (tInterp.AllocateTensors() != kTfLiteOk || rInterp.AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed (try a bigger ARENA_BYTES)");
    return false;
  }
  tempInterp = &tInterp;  riskInterp = &rInterp;
  tempIn = tInterp.input(0);  tempOut = tInterp.output(0);
  riskIn = rInterp.input(0);  riskOut = rInterp.output(0);
  return true;
}

inline int8_t quantizeValue(float x, float scale, int zp) {
  int q = (int)lroundf(x / scale) + zp;
  if (q < -128) q = -128;
  if (q > 127) q = 127;
  return (int8_t)q;
}

// window: WINDOW_SIZE * FEATURE_COUNT raw feature values, oldest sample first.
inline void fillInput(TfLiteTensor* in, const float* window, float scale, int zp) {
  for (int i = 0; i < INPUT_LEN; i++) {
    int f = i % FEATURE_COUNT;
    float norm = (window[i] - FEAT_MEAN[f]) / FEAT_STD[f];
    in->data.int8[i] = quantizeValue(norm, scale, zp);
  }
}

// Temperature forecast HORIZON_STEPS * STEP_MINUTES minutes ahead (C).
inline float predictTemperatureC(const float* window, uint32_t* micros_out) {
  fillInput(tempIn, window, TEMP_IN_SCALE, TEMP_IN_ZP);
  uint32_t t0 = micros();
  if (tempInterp->Invoke() != kTfLiteOk) return NAN;
  if (micros_out) *micros_out = micros() - t0;
  float norm = (tempOut->data.int8[0] - TEMP_OUT_ZP) * TEMP_OUT_SCALE;
  return norm * TARGET_STD + TARGET_MEAN;
}

// Returns 0 low, 1 moderate, 2 high; -1 on error.
inline int predictRiskClass(const float* window, uint32_t* micros_out) {
  fillInput(riskIn, window, RISK_IN_SCALE, RISK_IN_ZP);
  uint32_t t0 = micros();
  if (riskInterp->Invoke() != kTfLiteOk) return -1;
  if (micros_out) *micros_out = micros() - t0;
  int best = 0;
  for (int c = 1; c < NUM_RISK_CLASSES; c++)
    if (riskOut->data.int8[c] > riskOut->data.int8[best]) best = c;
  return best;
}

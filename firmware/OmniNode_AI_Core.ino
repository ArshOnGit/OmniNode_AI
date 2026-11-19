/**
 * @file OmniNode_AI_Core.ino
 * @brief Multipurpose IoT Monitoring Node (OmniNode-AI) with Dual TinyML Models & RAM Rolling Buffer
 * @details Reads DHT22, calculates Magnus-Tetens Dew Point, NOAA Heat Index,
 *          runs FNN Forecasting, State Classification, and manages a RAM circular buffer for Blynk.
 * @author Arsh
 * @date October 2026
 */

#include <Arduino.h>
#include <DHT.h>
#include "model_config.h"
#include "environmental_features.h"
#include "sensor_processing.h"
#include "inference_engine.h"

// ---------- wiring (change if yours differs) ----------
constexpr uint8_t DHT_PIN = 4;
constexpr uint8_t BUZZER_PIN = 18;
constexpr uint8_t RELAY_PIN = 19;          // fan relay
constexpr bool RELAY_ACTIVE_LOW = false;   // many relay modules are active-LOW

// ---------- thresholds ----------
constexpr float TEMP_HIGH_C = 25.0f;
constexpr float TEMP_LOW_C = 18.0f;

// ---------- timing ----------
constexpr uint32_t READ_INTERVAL_MS = 30000UL;                    // DHT22 needs >= 2 s
constexpr uint32_t STEP_MS = (uint32_t)STEP_MINUTES * 60000UL;    // one model sample

DHT dht(DHT_PIN, DHT22);
StepAverager averager;

float history[WINDOW_SIZE][FEATURE_COUNT];
int histCount = 0;
int histHead = 0;   // next write position

bool modelsReady = false;
float lastForecastC = NAN;
int lastRisk = -1;
uint32_t lastTempInferUs = 0, lastRiskInferUs = 0;
uint32_t lastRead = 0, stepStart = 0;

void setOutputs(bool buzzer, bool fan) {
  digitalWrite(BUZZER_PIN, buzzer ? HIGH : LOW);
  bool level = RELAY_ACTIVE_LOW ? !fan : fan;
  digitalWrite(RELAY_PIN, level ? HIGH : LOW);
}

void pushSample(const Features& f) {
  history[histHead][0] = f.t;
  history[histHead][1] = f.rh;
  history[histHead][2] = f.dew;
  history[histHead][3] = f.dpd;
  history[histHead][4] = f.vpd;
  histHead = (histHead + 1) % WINDOW_SIZE;
  if (histCount < WINDOW_SIZE) histCount++;
}

void copyWindow(float* out) {   // oldest first; call only when histCount == WINDOW_SIZE
  int start = histHead;
  for (int i = 0; i < WINDOW_SIZE; i++) {
    int idx = (start + i) % WINDOW_SIZE;
    memcpy(out + i * FEATURE_COUNT, history[idx], FEATURE_COUNT * sizeof(float));
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(RELAY_PIN, OUTPUT);
  setOutputs(false, false);
  dht.begin();
  modelsReady = initInference();
  Serial.println(modelsReady ? "Models ready" : "Models NOT ready - using rule-based risk only");
  delay(2000);
  lastRead = millis() - READ_INTERVAL_MS;   // read immediately in loop()
  stepStart = millis();
}

void loop() {
  uint32_t now = millis();
  if (now - lastRead < READ_INTERVAL_MS) return;
  lastRead = now;

  SensorReading r = readSensor(dht);
  if (!r.valid) {
    Serial.println("Sensor read failed - skipping");
    return;
  }
  Features f = computeFeatures(r.tempC, r.rh);
  averager.add(r.tempC, r.rh);

  // every STEP_MS: push one 5-minute averaged sample and run the models
  if (now - stepStart >= STEP_MS && averager.count > 0) {
    pushSample(computeFeatures(averager.meanT(), averager.meanH()));
    averager.reset();
    stepStart = now;
    if (modelsReady && histCount == WINDOW_SIZE) {
      float window[INPUT_LEN];
      copyWindow(window);
      lastForecastC = predictTemperatureC(window, &lastTempInferUs);
      lastRisk = predictRiskClass(window, &lastRiskInferUs);
    }
  }

  int risk = (lastRisk >= 0) ? lastRisk : ruleRiskClass(f);
  bool tempAlert = f.t > TEMP_HIGH_C || f.t < TEMP_LOW_C;
  bool forecastAlert = !isnan(lastForecastC) &&
                       (lastForecastC > TEMP_HIGH_C || lastForecastC < TEMP_LOW_C);
  bool riskHigh = (risk == 2);
  setOutputs(tempAlert || forecastAlert || riskHigh, f.t > TEMP_HIGH_C || riskHigh);

  Serial.printf("T=%.1fC RH=%.1f%% Dew=%.1fC DPD=%.1f VPD=%.2fkPa | forecast30m=%.1fC risk=%d | "
                "alerts: temp=%d forecast=%d risk=%d | window=%d/%d | infer_us=%lu/%lu heap=%u\n",
                f.t, f.rh, f.dew, f.dpd, f.vpd, lastForecastC, risk,
                tempAlert, forecastAlert, riskHigh, histCount, WINDOW_SIZE,
                (unsigned long)lastTempInferUs, (unsigned long)lastRiskInferUs,
                ESP.getFreeHeap());
}

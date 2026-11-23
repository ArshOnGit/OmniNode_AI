#include <Arduino.h>
#include <DHT.h>
#include <math.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>

#include "environmental_warning.h"
#include "omninode_forecaster_v2_int8.h"
#include "watanabe_risk.h"

constexpr uint8_t kDhtPin = 4;
constexpr uint32_t kSampleIntervalMs = 60000UL;
constexpr uint32_t kMinuteMs = 60000UL;
constexpr uint32_t kForecastIntervalMinutes = 5UL;
constexpr uint32_t kForecastSamples = 4;
constexpr uint32_t kTensorArenaBytes = 16UL * 1024UL;
constexpr size_t kHistoryCapacity = 60;
constexpr float kWatanabeWindowHours = 48.0f;

constexpr float kModelInputMean[8] = {
    19.725215911865234f,
    82.48782348632812f,
    0.0007819808670319617f,
    -0.0011214861879125237f,
    0.000536816893145442f,
    -0.0007958641508594155f,
    0.0002701958001125604f,
    -0.00043258751975372434f
};

constexpr float kModelInputScale[8] = {
    3.3781964778900146f,
    10.359574317932129f,
    0.4337501525878906f,
    1.8178669214248657f,
    0.3364019989967346f,
    1.487447738647461f,
    0.20738011598587036f,
    1.0465890169143677f
};

constexpr float kModelOutputMean[2] = {
    -0.001559551921673119f,
    0.0019722438883036375f
};

constexpr float kModelOutputScale[2] = {
    0.6615920066833496f,
    2.625882148742676f
};

constexpr float kModelInputQuantScale = 0.05126921087503433f;
constexpr int32_t kModelInputQuantZeroPoint = -17;

constexpr float kModelOutputQuantScale = 0.011703990399837494f;
constexpr int32_t kModelOutputQuantZeroPoint = 1;

constexpr float kQuantizationTolerance = 0.000001f;
constexpr float kModerateRiskMinHumidity = 85.0f;
constexpr float kHighRiskMinHumidity = 90.0f;
constexpr float kModerateRiskMinTemperatureC = 10.0f;
constexpr float kModerateRiskMaxTemperatureC = 30.0f;
constexpr float kHighRiskMinTemperatureC = 15.0f;
constexpr float kHighRiskMaxTemperatureC = 25.0f;

struct EnvironmentSample {
  float temperatureC;
  float relativeHumidity;
  float dewPointC;
  float vpdKPa;
  float dewPointMarginC;
};

enum class EnvironmentalRisk : uint8_t {
  Low,
  Moderate,
  High,
};

DHT dht(kDhtPin, DHT22);
watanabe::Accumulator infectionIndex;  // 2-day Watanabe (2021) IAT
EnvironmentSample history[kHistoryCapacity] = {};
EnvironmentSample forecastWindow[kForecastSamples] = {};
size_t historyHead = 0;
size_t historyCount = 0;
size_t forecastCount = 0;
uint32_t lastSampleMs = 0;
uint32_t bootTimeMs = 0;
uint32_t lastForecastMinuteSlot = UINT32_MAX;
bool firstSamplePending = true;
bool forecastModelReady = false;

alignas(16) uint8_t tensorArena[kTensorArenaBytes];
tflite::MicroMutableOpResolver<1> opResolver;
tflite::MicroInterpreter interpreter(
    tflite::GetModel(g_omninode_forecaster_v2_int8_model), opResolver, tensorArena,
    sizeof(tensorArena));
TfLiteTensor* modelInput = nullptr;
TfLiteTensor* modelOutput = nullptr;

bool readEnvironmentSample(EnvironmentSample& sample) {
  const float temperatureC = dht.readTemperature();
  const float relativeHumidity = dht.readHumidity();

  if (!isfinite(temperatureC) || !isfinite(relativeHumidity)) {
    Serial.println("[ERROR] DHT22 returned a non-finite value; sample skipped.");
    return false;
  }

  if (temperatureC < -40.0f || temperatureC > 80.0f ||
      relativeHumidity <= 0.0f || relativeHumidity > 100.0f) {
    Serial.printf(
        "[ERROR] DHT22 reading out of range (T=%.2f C, RH=%.2f%%); sample skipped.\n",
        temperatureC, relativeHumidity);
    return false;
  }

  environmental_warning::Metrics metrics = {};
  if (!environmental_warning::calculateMetrics(
          temperatureC, relativeHumidity, metrics)) {
    Serial.println("[ERROR] Environmental feature calculation failed; sample skipped.");
    return false;
  }

  sample.temperatureC = temperatureC;
  sample.relativeHumidity = relativeHumidity;
  sample.dewPointC = metrics.dewPointC;
  sample.vpdKPa = metrics.vpdKPa;
  sample.dewPointMarginC = metrics.dewPointMarginC;
  return true;
}

EnvironmentalRisk estimateEnvironmentalRisk(
    const EnvironmentSample& sample) {
  const bool moderateTemperature =
      sample.temperatureC >= kModerateRiskMinTemperatureC &&
      sample.temperatureC <= kModerateRiskMaxTemperatureC;
  const bool highTemperature =
      sample.temperatureC >= kHighRiskMinTemperatureC &&
      sample.temperatureC <= kHighRiskMaxTemperatureC;

  if (sample.relativeHumidity >= kHighRiskMinHumidity && highTemperature) {
    return EnvironmentalRisk::High;
  }
  if (sample.relativeHumidity >= kModerateRiskMinHumidity &&
      moderateTemperature) {
    return EnvironmentalRisk::Moderate;
  }
  return EnvironmentalRisk::Low;
}

const char* riskName(EnvironmentalRisk risk) {
  switch (risk) {
    case EnvironmentalRisk::Moderate:
      return "MODERATE";
    case EnvironmentalRisk::High:
      return "HIGH";
    case EnvironmentalRisk::Low:
    default:
      return "LOW";
  }
}

void storeSample(const EnvironmentSample& sample) {
  history[historyHead] = sample;
  historyHead = (historyHead + 1) % kHistoryCapacity;
  if (historyCount < kHistoryCapacity) {
    ++historyCount;
  }
}

bool registerForecastOperators() {
  if (opResolver.AddFullyConnected() != kTfLiteOk) {
    Serial.println("[ERROR] Could not register required TFLite Micro operators.");
    return false;
  }
  return true;
}

bool hasExpectedShape(const TfLiteTensor* tensor, const int32_t* dimensions,
                      int dimensionCount) {
  if (tensor == nullptr || tensor->dims == nullptr ||
      tensor->dims->size != dimensionCount) {
    return false;
  }
  for (int i = 0; i < dimensionCount; ++i) {
    if (tensor->dims->data[i] != dimensions[i]) {
      return false;
    }
  }
  return true;
}

bool initializeForecastModel() {
  const tflite::Model* model =
      tflite::GetModel(g_omninode_forecaster_v2_int8_model);
  if (model == nullptr || model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("[ERROR] Forecast model schema is invalid or unsupported.");
    return false;
  }
  if (!registerForecastOperators()) {
    return false;
  }
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    Serial.println(
        "[ERROR] Forecast tensor allocation failed; monitoring will continue.");
    return false;
  }

  modelInput = interpreter.input(0);
  modelOutput = interpreter.output(0);
  const int32_t expectedInputDimensions[] = {1, 8};
  const int32_t expectedOutputDimensions[] = {1, 2};
  if (!hasExpectedShape(modelInput, expectedInputDimensions, 2) ||
      !hasExpectedShape(modelOutput, expectedOutputDimensions, 2) ||
      modelInput->type != kTfLiteInt8 || modelOutput->type != kTfLiteInt8) {
    Serial.println(
        "[ERROR] Forecast model tensor shape/type differs from its metadata.");
    return false;
  }

  if (fabsf(modelInput->params.scale - kModelInputQuantScale) >
          kQuantizationTolerance ||
      modelInput->params.zero_point != kModelInputQuantZeroPoint ||
      fabsf(modelOutput->params.scale - kModelOutputQuantScale) >
          kQuantizationTolerance ||
      modelOutput->params.zero_point != kModelOutputQuantZeroPoint) {
    Serial.println(
        "[ERROR] Forecast model quantization differs from its metadata.");
    return false;
  }

  forecastModelReady = true;
  Serial.printf(
      "[INFO] INT8 forecaster ready; tensor arena used: %u/%u bytes.\n",
      static_cast<unsigned>(interpreter.arena_used_bytes()),
      static_cast<unsigned>(sizeof(tensorArena)));
  return true;
}

void appendForecastSample(const EnvironmentSample& sample,
                          uint32_t minuteSlot) {
  if (lastForecastMinuteSlot != UINT32_MAX &&
      minuteSlot - lastForecastMinuteSlot != kForecastIntervalMinutes) {
    forecastCount = 0;
    Serial.println(
        "[INFO] Forecast window reset because a five-minute sample was missed.");
  }
  lastForecastMinuteSlot = minuteSlot;

  if (forecastCount < kForecastSamples) {
    forecastWindow[forecastCount++] = sample;
  } else {
    for (size_t i = 1; i < kForecastSamples; ++i) {
      forecastWindow[i - 1] = forecastWindow[i];
    }
    forecastWindow[kForecastSamples - 1] = sample;
  }
}

int8_t quantizeInt8(float value, float scale, int32_t zeroPoint) {
  long quantized = lroundf(value / scale) + zeroPoint;
  if (quantized < -128L) {
    quantized = -128L;
  } else if (quantized > 127L) {
    quantized = 127L;
  }
  return static_cast<int8_t>(quantized);
}

void runForecast() {
  if (!forecastModelReady || modelInput == nullptr || modelOutput == nullptr) {
    Serial.println("[ERROR] Forecast model unavailable; monitoring continues.");
    return;
  }

  const EnvironmentSample& oldest = forecastWindow[0];
  const EnvironmentSample& middle = forecastWindow[1];
  const EnvironmentSample& newestHistory = forecastWindow[2];
  const EnvironmentSample& latest = forecastWindow[kForecastSamples - 1];
  const float latestTemperature = latest.temperatureC;
  const float latestHumidity = latest.relativeHumidity;
  const float features[8] = {
      latestTemperature,
      latestHumidity,
      oldest.temperatureC - latestTemperature,
      oldest.relativeHumidity - latestHumidity,
      middle.temperatureC - latestTemperature,
      middle.relativeHumidity - latestHumidity,
      newestHistory.temperatureC - latestTemperature,
      newestHistory.relativeHumidity - latestHumidity,
  };
  for (size_t featureIndex = 0; featureIndex < 8; ++featureIndex) {
    const float normalized =
        (features[featureIndex] - kModelInputMean[featureIndex]) /
        kModelInputScale[featureIndex];
    modelInput->data.int8[featureIndex] =
        quantizeInt8(normalized, kModelInputQuantScale,
                     kModelInputQuantZeroPoint);
  }

  const uint32_t inferenceStartUs = micros();
  if (interpreter.Invoke() != kTfLiteOk) {
    Serial.println("[ERROR] TFLite Micro forecast inference failed.");
    return;
  }
  const uint32_t inferenceTimeUs = micros() - inferenceStartUs;

  const float predictedTemperatureDelta =
      (static_cast<int32_t>(modelOutput->data.int8[0]) -
       kModelOutputQuantZeroPoint) *
          kModelOutputQuantScale * kModelOutputScale[0] +
      kModelOutputMean[0];
  const float predictedHumidityDelta =
      (static_cast<int32_t>(modelOutput->data.int8[1]) -
       kModelOutputQuantZeroPoint) *
          kModelOutputQuantScale * kModelOutputScale[1] +
      kModelOutputMean[1];
  const float predictedTemperature =
      latestTemperature + predictedTemperatureDelta;
  const float predictedHumidity = latestHumidity + predictedHumidityDelta;

  if (!isfinite(predictedTemperature) || !isfinite(predictedHumidity)) {
    Serial.println("[ERROR] Forecast model returned a non-finite prediction.");
    return;
  }

  environmental_warning::Metrics predictedMetrics = {};
  if (!environmental_warning::calculateMetrics(
          predictedTemperature, predictedHumidity, predictedMetrics)) {
    Serial.println(
        "[ERROR] Predicted environmental metrics are invalid; forecast warning unavailable.");
    return;
  }
  const environmental_warning::Level predictedWarning =
      environmental_warning::classify(
          predictedTemperature, predictedHumidity, predictedMetrics);

  Serial.printf(
      "[FORECAST] +30 min: temperature %.2f C, relative humidity %.2f %% "
      "(environmental warning: %s; inference %u us).\n",
      predictedTemperature, predictedHumidity,
      environmental_warning::levelName(predictedWarning),
      static_cast<unsigned>(inferenceTimeUs));
  Serial.printf(
      "@JSON {\"type\":\"forecast\",\"horizon_min\":30,"
      "\"temperature_c\":%.2f,\"relative_humidity_pct\":%.2f,"
      "\"inference_us\":%u,\"predicted_vpd_kpa\":%.3f,"
      "\"predicted_dew_point_c\":%.2f,"
      "\"predicted_dew_point_margin_c\":%.2f,"
      "\"predicted_environmental_warning\":\"%s\","
      "\"warning_horizon_min\":30,\"warning_basis\":\"%s\"}\n",
      predictedTemperature, predictedHumidity,
      static_cast<unsigned>(inferenceTimeUs), predictedMetrics.vpdKPa,
      predictedMetrics.dewPointC, predictedMetrics.dewPointMarginC,
      environmental_warning::levelName(predictedWarning),
      environmental_warning::kBasis);
}

// Machine-readable telemetry for the planned custom dashboard. One JSON object
// per line, prefixed with "@JSON " so a host bridge can separate it from logs.
void emitSampleTelemetry(const EnvironmentSample& sample) {
  const environmental_warning::Metrics metrics = {
      sample.dewPointC, sample.vpdKPa, sample.dewPointMarginC};
  const environmental_warning::Level warning =
      environmental_warning::classify(
          sample.temperatureC, sample.relativeHumidity, metrics);
  const bool iatWindowComplete =
      infectionIndex.coverageHours() >= kWatanabeWindowHours;
  Serial.printf(
      "@JSON {\"type\":\"sample\",\"uptime_s\":%lu,\"temperature_c\":%.2f,"
      "\"relative_humidity_pct\":%.2f,\"dew_point_c\":%.2f,\"vpd_kpa\":%.3f,"
      "\"dew_point_margin_c\":%.2f,\"risk\":\"%s\",\"risk_basis\":\"watanabe_2021_iat\","
      "\"iat_hours\":%.2f,\"iat_coverage_hours\":%.2f,"
      "\"simple_rule_risk\":\"%s\",\"forecast_window\":%u,"
      "\"iat_window_complete\":%s,\"iat_warming_up\":%s,"
      "\"current_environmental_warning\":\"%s\",\"warning_basis\":\"%s\"}\n",
      static_cast<unsigned long>((millis() - bootTimeMs) / 1000UL),
      sample.temperatureC, sample.relativeHumidity, sample.dewPointC,
      sample.vpdKPa, sample.dewPointMarginC,
      watanabe::levelName(infectionIndex.level()), infectionIndex.hours(),
      infectionIndex.coverageHours(),
      riskName(estimateEnvironmentalRisk(sample)),
      static_cast<unsigned>(forecastCount),
      iatWindowComplete ? "true" : "false",
      iatWindowComplete ? "false" : "true",
      environmental_warning::levelName(warning),
      environmental_warning::kBasis);
}

void printSample(const EnvironmentSample& sample) {
  Serial.println("----------------------------------------");
  Serial.printf("Temperature:       %.2f C\n", sample.temperatureC);
  Serial.printf("Relative humidity: %.2f %%\n", sample.relativeHumidity);
  Serial.printf("Dew point:         %.2f C\n", sample.dewPointC);
  Serial.printf("VPD:               %.3f kPa\n", sample.vpdKPa);
  Serial.printf("Dew-point margin:  %.2f C\n", sample.dewPointMarginC);
  Serial.printf(
      "Botrytis infection-risk index (Watanabe 2021): %s, %.1f h effective in "
      "last 2 days (%.1f h covered%s)\n",
      watanabe::levelName(infectionIndex.level()), infectionIndex.hours(),
      infectionIndex.coverageHours(),
      infectionIndex.coverageHours() >= kWatanabeWindowHours
          ? ""
          : "; lower bound, window warming up");
  Serial.printf("Simple point-in-time rule (legacy): %s\n",
                riskName(estimateEnvironmentalRisk(sample)));
  const environmental_warning::Metrics metrics = {
      sample.dewPointC, sample.vpdKPa, sample.dewPointMarginC};
  Serial.printf(
      "Current environmental warning (demo): %s\n",
      environmental_warning::levelName(environmental_warning::classify(
          sample.temperatureC, sample.relativeHumidity, metrics)));
  Serial.printf("History:           %u/%u one-minute samples\n",
                static_cast<unsigned>(historyCount),
                static_cast<unsigned>(kHistoryCapacity));
  Serial.printf("Forecast window:   %s (%u/%u five-minute samples)\n",
                forecastCount == kForecastSamples ? "ready" : "warming up",
                static_cast<unsigned>(forecastCount),
                static_cast<unsigned>(kForecastSamples));
}

void setup() {
  Serial.begin(115200);
  dht.begin();
  bootTimeMs = millis();

  Serial.println("\n[INFO] OmniNode-AI offline greenhouse monitoring started.");
  Serial.println(
      "[INFO] Risk labels are provisional environmental indicators, not "
      "infection detection.");
  initializeForecastModel();
}

void loop() {
  const uint32_t now = millis();
  if (!firstSamplePending &&
      static_cast<uint32_t>(now - lastSampleMs) < kSampleIntervalMs) {
    return;
  }

  firstSamplePending = false;
  lastSampleMs = now;
  const uint32_t minuteSlot =
      static_cast<uint32_t>(now - bootTimeMs) / kMinuteMs;
  const bool forecastSampleDue =
      minuteSlot % kForecastIntervalMinutes == 0 &&
      minuteSlot != lastForecastMinuteSlot;

  EnvironmentSample sample = {};
  const bool sampleValid = readEnvironmentSample(sample);
  if (forecastSampleDue) {
    if (sampleValid) {
      appendForecastSample(sample, minuteSlot);
    } else {
      lastForecastMinuteSlot = minuteSlot;
      forecastCount = 0;
      Serial.println(
          "[ERROR] Invalid five-minute reading; forecast window cleared.");
    }
  }

  if (!sampleValid) {
    return;
  }

  storeSample(sample);
  infectionIndex.addMinute(minuteSlot, sample.temperatureC,
                           sample.relativeHumidity);
  printSample(sample);
  emitSampleTelemetry(sample);

  if (forecastSampleDue && forecastCount == kForecastSamples) {
    runForecast();
  }
}

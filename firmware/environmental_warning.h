#pragma once

#include <math.h>
#include <stdint.h>

namespace environmental_warning {

struct Metrics {
  float dewPointC;
  float vpdKPa;
  float dewPointMarginC;
};

enum class Level : uint8_t {
  Low,
  Caution,
  High,
};

constexpr float kMagnusA = 17.27f;
constexpr float kMagnusB = 237.7f;
constexpr float kCautionMinTemperatureC = 10.0f;
constexpr float kCautionMaxTemperatureC = 30.0f;
constexpr float kCautionMinHumidity = 85.0f;
constexpr float kCautionMaxVpdKPa = 0.80f;
constexpr float kCautionMaxDewPointMarginC = 3.5f;
constexpr float kHighMinTemperatureC = 15.0f;
constexpr float kHighMaxTemperatureC = 25.0f;
constexpr float kHighMinHumidity = 90.0f;
constexpr float kHighMaxVpdKPa = 0.45f;
constexpr float kHighMaxDewPointMarginC = 2.1f;
constexpr char kBasis[] = "configurable_demo_thresholds_v1";

inline bool calculateMetrics(float temperatureC, float relativeHumidity,
                             Metrics& metrics) {
  if (!isfinite(temperatureC) || !isfinite(relativeHumidity) ||
      temperatureC < -40.0f || temperatureC > 80.0f ||
      relativeHumidity <= 0.0f || relativeHumidity > 100.0f) {
    return false;
  }

  const float alpha =
      logf(relativeHumidity / 100.0f) +
      kMagnusA * temperatureC / (kMagnusB + temperatureC);
  const float dewPointC = kMagnusB * alpha / (kMagnusA - alpha);
  const float saturationVaporPressureKPa =
      0.6108f * expf(kMagnusA * temperatureC / (kMagnusB + temperatureC));
  const float actualVaporPressureKPa =
      saturationVaporPressureKPa * relativeHumidity / 100.0f;
  const float vpdKPa =
      saturationVaporPressureKPa - actualVaporPressureKPa;

  if (!isfinite(dewPointC) || !isfinite(vpdKPa) || vpdKPa < 0.0f) {
    return false;
  }

  metrics.dewPointC = dewPointC;
  metrics.vpdKPa = vpdKPa;
  metrics.dewPointMarginC = temperatureC - dewPointC;
  return true;
}

inline Level classify(float temperatureC, float relativeHumidity,
                      const Metrics& metrics) {
  if (temperatureC >= kHighMinTemperatureC &&
      temperatureC <= kHighMaxTemperatureC &&
      relativeHumidity >= kHighMinHumidity &&
      metrics.vpdKPa <= kHighMaxVpdKPa &&
      metrics.dewPointMarginC <= kHighMaxDewPointMarginC) {
    return Level::High;
  }
  if (temperatureC >= kCautionMinTemperatureC &&
      temperatureC <= kCautionMaxTemperatureC &&
      relativeHumidity >= kCautionMinHumidity &&
      metrics.vpdKPa <= kCautionMaxVpdKPa &&
      metrics.dewPointMarginC <= kCautionMaxDewPointMarginC) {
    return Level::Caution;
  }
  return Level::Low;
}

inline const char* levelName(Level level) {
  switch (level) {
    case Level::Caution:
      return "CAUTION";
    case Level::High:
      return "HIGH";
    case Level::Low:
    default:
      return "LOW";
  }
}

}  // namespace environmental_warning

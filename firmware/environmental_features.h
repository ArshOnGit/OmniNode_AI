#pragma once
#include <math.h>

constexpr float MT_A = 17.27f;
constexpr float MT_B = 237.7f;

struct Features {
  float t;    // temperature, C
  float rh;   // relative humidity, %
  float dew;  // dew point, C
  float dpd;  // dew point delta = t - dew
  float vpd;  // vapour pressure deficit, kPa
};

inline float dewPointC(float t, float rh) {
  float g = MT_A * t / (MT_B + t) + logf(rh / 100.0f);
  return MT_B * g / (MT_A - g);
}

inline float vpdKpa(float t, float rh) {
  float es = 0.6108f * expf(17.27f * t / (t + 237.3f));
  return es * (1.0f - rh / 100.0f);
}

inline Features computeFeatures(float t, float rh) {
  Features f;
  f.t = t;
  f.rh = rh;
  f.dew = dewPointC(t, rh);
  f.dpd = t - f.dew;
  f.vpd = vpdKpa(t, rh);
  return f;
}

// Rule-based risk (same thresholds as data_preprocessing.py): 0 low, 1 moderate, 2 high
inline int ruleRiskClass(const Features& f) {
  if (f.dpd <= 1.0f || f.rh >= 90.0f) return 2;
  if (f.dpd <= 2.5f || f.rh >= 85.0f || f.vpd < 0.4f) return 1;
  return 0;
}

// Watanabe et al. (2021) cumulative effective-infection-time index for
// Botrytis cinerea on greenhouse tomato. Pure C++ (no Arduino dependencies) so
// it can be unit-tested on a host. Ann. Rept. Kansai Pl. Prot. 63: 59-65,
// doi:10.4165/kapps.63.59.
//
// Method (paper eq. 1-5):
//   * A logging interval counts as "infectable" when air temperature minus dew
//     point is < 1.2 C (about RH >= 93-94 %).
//   * Each infectable interval is weighted by Y(T), a 6th-order polynomial fit
//     of relative mycelial growth vs temperature (1.0 at the 22.5 C optimum).
//   * IAT = sum of weighted hours over the previous 2 days.
//   * IAT < 15 h: safety; 15 <= IAT < 20 h: caution; IAT >= 20 h: danger.
//
// Deviations / assumptions (ours, not the paper's):
//   * Y is clamped to [0, 1] and set to 0 outside 3-30 C (the paper measured
//     growth only over 3-30 C; the polynomial is meaningless outside it and
//     goes negative above ~29 C). The paper reports no growth at 30 C.
//   * The paper's polynomial gives Y(28 C) ~= 0.26 although the measured ratio
//     was 0.50; we use the published polynomial unchanged.
//   * The device has no real-time clock, so time is minutes since boot. Until
//     48 h of coverage have been collected the index is a lower bound.
//   * Thresholds were derived on one greenhouse/cultivar; treat as indicative.
#pragma once
#include <math.h>
#include <stddef.h>
#include <stdint.h>

namespace watanabe {

constexpr size_t kWindowMinutes = 2 * 24 * 60;  // previous 2 days
constexpr float kDewPointMarginThresholdC = 1.2f;
constexpr float kCautionHours = 15.0f;
constexpr float kDangerHours = 20.0f;

enum class Level : uint8_t { Safety, Caution, Danger };

inline const char* levelName(Level l) {
  return l == Level::Danger ? "DANGER" : l == Level::Caution ? "CAUTION" : "SAFETY";
}

// Dew point using the paper's constants (eq. 1-3), hPa-based.
inline float dewPointC(float tempC, float rh) {
  const float es = 6.1078f * expf(17.2693882f * tempC / (tempC + 237.3f));
  const float e = rh * es / 100.0f;
  return 237.3f * logf(e / 6.1078f) / (17.2693882f - logf(e / 6.1078f));
}

// Mycelial growth adjustment factor Y(T), paper eq. 5, evaluated in double.
inline float growthFactor(float tempC) {
  if (!(tempC >= 3.0f && tempC <= 30.0f)) return 0.0f;
  const double t = tempC;
  double y = 0.00000013611;
  y = y * t - 0.00001325037;
  y = y * t + 0.00049243766;
  y = y * t - 0.00900511846;
  y = y * t + 0.083895;
  y = y * t - 0.3035;
  y = y * t + 0.436;
  if (y < 0.0) y = 0.0;
  if (y > 1.0) y = 1.0;
  return static_cast<float>(y);
}

// Effective weight (0..1) of one logging interval for given T and RH.
inline float intervalWeight(float tempC, float rh) {
  if (!(rh > 0.0f) || rh > 100.0f) return 0.0f;
  if (tempC - dewPointC(tempC, rh) >= kDewPointMarginThresholdC) return 0.0f;
  return growthFactor(tempC);
}

// Sliding 2-day accumulator at one-minute resolution. Minutes with no valid
// reading contribute 0 and are tracked in `coveredMinutes`.
class Accumulator {
 public:
  // Record a valid reading for absolute minute `minute` (monotonic, since boot).
  void addMinute(uint32_t minute, float tempC, float rh) {
    advanceTo(minute);
    const uint8_t q = static_cast<uint8_t>(lroundf(intervalWeight(tempC, rh) * 255.0f));
    const size_t slot = minute % kWindowMinutes;
    if (!valid_[slot]) { valid_[slot] = true; ++covered_; }
    sumQ_ += q - weights_[slot];
    weights_[slot] = q;
  }
  // Advance time without a reading (clears expired slots).
  void advanceTo(uint32_t minute) {
    if (!started_) { started_ = true; head_ = minute; return; }
    if (minute <= head_) return;
    uint32_t gap = minute - head_;
    if (gap > kWindowMinutes) gap = kWindowMinutes;
    for (uint32_t i = 1; i <= gap; ++i) clear((head_ + i) % kWindowMinutes);
    head_ = minute;
  }
  float hours() const { return static_cast<float>(sumQ_) / 255.0f / 60.0f; }
  float coverageHours() const { return static_cast<float>(covered_) / 60.0f; }
  Level level() const {
    const float h = hours();
    return h >= kDangerHours ? Level::Danger : h >= kCautionHours ? Level::Caution : Level::Safety;
  }
  bool windowFull() const { return covered_ >= kWindowMinutes * 9 / 10; }

 private:
  void clear(size_t slot) {
    if (valid_[slot]) { valid_[slot] = false; --covered_; }
    sumQ_ -= weights_[slot];
    weights_[slot] = 0;
  }
  uint8_t weights_[kWindowMinutes] = {};
  bool valid_[kWindowMinutes] = {};
  int32_t sumQ_ = 0;
  uint32_t covered_ = 0;
  uint32_t head_ = 0;
  bool started_ = false;
};

}  // namespace watanabe

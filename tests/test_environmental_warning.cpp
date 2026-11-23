#include <assert.h>
#include <math.h>

#include "environmental_warning.h"

int main() {
  environmental_warning::Metrics metrics = {};

  assert(environmental_warning::calculateMetrics(28.5f, 69.8f, metrics));
  assert(fabsf(metrics.dewPointC - 22.44f) < 0.02f);
  assert(fabsf(metrics.vpdKPa - 1.172f) < 0.01f);
  assert(fabsf(metrics.dewPointMarginC - 6.06f) < 0.02f);
  assert(environmental_warning::classify(28.5f, 69.8f, metrics) ==
         environmental_warning::Level::Low);

  assert(environmental_warning::calculateMetrics(22.0f, 88.0f, metrics));
  assert(environmental_warning::classify(22.0f, 88.0f, metrics) ==
         environmental_warning::Level::Caution);

  assert(environmental_warning::calculateMetrics(22.0f, 95.0f, metrics));
  assert(environmental_warning::classify(22.0f, 95.0f, metrics) ==
         environmental_warning::Level::High);

  assert(!environmental_warning::calculateMetrics(22.0f, 0.0f, metrics));
  assert(!environmental_warning::calculateMetrics(22.0f, 100.1f, metrics));
  assert(!environmental_warning::calculateMetrics(NAN, 80.0f, metrics));
}

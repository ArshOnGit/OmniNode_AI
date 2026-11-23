// Host test: prints accumulator results for a scripted scenario so the Python
// reference can be compared. Build: g++ -std=c++17 -Ifirmware tests/test_watanabe.cpp
#include <cstdio>
#include "watanabe_risk.h"
int main() {
  static watanabe::Accumulator acc;
  // 30 h at 22.5C / 97% RH, then 30 h dry, then 10 h wet again.
  uint32_t m = 0;
  for (int i = 0; i < 30 * 60; ++i, ++m) acc.addMinute(m, 22.5f, 97.0f);
  printf("A %.3f %s\n", acc.hours(), watanabe::levelName(acc.level()));
  for (int i = 0; i < 30 * 60; ++i, ++m) acc.addMinute(m, 22.5f, 60.0f);
  printf("B %.3f %s\n", acc.hours(), watanabe::levelName(acc.level()));
  for (int i = 0; i < 10 * 60; ++i, ++m) acc.addMinute(m, 15.0f, 97.0f);
  printf("C %.3f %s\n", acc.hours(), watanabe::levelName(acc.level()));
  printf("Y %.4f %.4f %.4f %.4f\n", watanabe::growthFactor(10), watanabe::growthFactor(22.5f),
         watanabe::growthFactor(28), watanabe::growthFactor(31));
  printf("Dp %.3f\n", watanabe::dewPointC(22.5f, 97.0f));
}

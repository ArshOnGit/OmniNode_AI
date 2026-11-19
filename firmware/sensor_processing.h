#pragma once
#include <Arduino.h>
#include <DHT.h>

struct SensorReading {
  float tempC;
  float rh;
  bool valid;
};

inline SensorReading readSensor(DHT& dht) {
  SensorReading r;
  r.tempC = dht.readTemperature();
  r.rh = dht.readHumidity();
  r.valid = !isnan(r.tempC) && !isnan(r.rh) &&
            r.tempC > -10.0f && r.tempC < 60.0f &&
            r.rh >= 1.0f && r.rh <= 100.0f;
  return r;
}

struct StepAverager {
  double sumT = 0, sumH = 0;
  uint16_t count = 0;
  void add(float t, float h) { sumT += t; sumH += h; count++; }
  float meanT() const { return count ? (float)(sumT / count) : NAN; }
  float meanH() const { return count ? (float)(sumH / count) : NAN; }
  void reset() { sumT = 0; sumH = 0; count = 0; }
};

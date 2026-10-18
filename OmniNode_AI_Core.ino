/**
 * @file OmniNode_Sensor_Engine.ino
 * @brief Multipurpose IoT Monitoring Node (OmniNode-AI) - Firmware Core
 * @details Reads temperature and humidity from DHT22, computes Magnus-Tetens 
 *          Dew Point and NOAA Heat Index locally on ESP32.
 * @author Arsh
 * @date October 2026
 */

#include <DHT.h>

// Define pin and sensor type
#define DHTPIN 4     // Digital pin connected to the DHT data pin (GPIO 4)
#define DHTTYPE DHT22   // Using DHT 22 (AM2302), change to DHT11 if using DHT11

DHT dht(DHTPIN, DHTTYPE);

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("OmniNode-AI: Initializing Sensor & Calculations...");
  
  dht.begin();
}

void loop() {
  // Wait 2 seconds between measurements
  delay(2000);

  // Read raw values
  float humidity = dht.readHumidity();
  float tempC = dht.readTemperature(); // Celsius

  // Check if readings failed
  if (isnan(humidity) || isnan(tempC)) {
    Serial.println("Error: Failed to read from DHT sensor!");
    return;
  }

  // 1. Calculate Dew Point using Magnus-Tetens Formula
  // Constants for water vapor in air over water (Tetens coefficients)
  float a = 17.27;
  float b = 237.7;
  float alpha = ((a * tempC) / (b + tempC)) + log(humidity / 100.0);
  float dewPoint = (b * alpha) / (a - alpha);

  // 2. Calculate Heat Index using NOAA Regression Equation (Fahrenheit first, then converted back or computed directly)
  // Converting Celsius to Fahrenheit for NOAA formula standard
  float tempF = (tempC * 9.0 / 5.0) + 32.0;
  
  // Simple sensible heat index formula check (Rothfusz regression)
  float heatIndexF = -42.379 + 2.04901523 * tempF + 10.14333127 * humidity
                   - 0.22475541 * tempF * humidity - 0.00683783 * tempF * tempF
                   - 0.05481717 * humidity * humidity + 0.00122874 * tempF * tempF * humidity
                   + 0.00085282 * tempF * humidity * humidity - 0.00000199 * tempF * tempF * humidity * humidity;
                   
  // Convert back to Celsius
  float heatIndexC = (heatIndexF - 32.0) * 5.0 / 9.0;

  // Print results cleanly to the Serial Monitor
  Serial.println("----------------------------------------");
  Serial.print("Ambient Temperature: "); Serial.print(tempC); Serial.println(" °C");
  Serial.print("Relative Humidity:   "); Serial.print(humidity); Serial.println(" %");
  Serial.print("Dew Point (Magnus):  "); Serial.print(dewPoint); Serial.println(" °C");
  Serial.print("Heat Index (NOAA):   "); Serial.print(heatIndexC); Serial.println(" °C");
}
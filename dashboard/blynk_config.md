# Blynk dashboard configuration

## Template
- Template name: OmniNode AI
- Hardware: ESP32, connection: Wi-Fi

## Datastreams
| Pin | Name              | Type   | Range / unit |
|-----|-------------------|--------|--------------|
| V0  | Temperature       | Double | -10 to 60 C  |
| V1  | Humidity          | Double | 0 to 100 %   |
| V2  | Dew point         | Double | -10 to 50 C  |
| V3  | Dew point delta   | Double | 0 to 30 C    |
| V4  | VPD               | Double | 0 to 5 kPa   |
| V5  | Forecast (30 min) | Double | -10 to 60 C  |
| V6  | Risk class        | Integer| 0 low, 1 moderate, 2 high |
| V7  | Alert             | Integer| 0 = ok, 1 = alert |

## Widgets
SuperChart (V0, V5), Gauges (V1, V4), Label/LED (V6, V7), Event notification for V7.

## Credentials (never commit)
Put these in `firmware/secrets.h`, which is gitignored:

    #define BLYNK_TEMPLATE_ID "xxxx"
    #define BLYNK_TEMPLATE_NAME "OmniNode AI"
    #define BLYNK_AUTH_TOKEN "xxxx"
    const char WIFI_SSID[] = "your-ssid";
    const char WIFI_PASS[] = "your-password";

## Offline behaviour
Sensing, inference, buzzer and relay run on the ESP32 without Wi-Fi.
Blynk updates resume when connectivity returns.

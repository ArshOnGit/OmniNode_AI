#!/usr/bin/env python3
"""Local bridge: ESP32 serial '@JSON ' telemetry -> HTTP for the dashboard.

  python dashboard/bridge.py --port COM3             # real board (needs pyserial)
  python dashboard/bridge.py --simulate              # fake data, no hardware
  python dashboard/bridge.py --replay saved-log.txt  # saved @JSON lines, no hardware

Every sample is stored in dashboard/history.sqlite (tagged device/simulated/replay);
/api/export.csv downloads it. The Watanabe (2021) index is recomputed from
stored history, so it survives device reboots. Simulated and replayed rows
never mix with device rows.

Serves index.html at http://localhost:8000 and the latest data at /api/state.
Simulated data is labelled as such in the API and shown in the UI.
"""
import argparse, json, math, random, sqlite3, sys, threading, time
import watanabe
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PREFIX = "@JSON "
state = {"source": "none", "demo_data": False, "samples": deque(maxlen=240), "all_samples": [],
         "forecasts": [], "forecast": None, "last_update": None,
         "last_raw_uptime": None, "last_session_uptime": None,
         "uptime_offset": 0.0, "last_sample_uptime": None}
lock = threading.Lock()
DB_PATH = Path(__file__).parent / "history.sqlite"
db = None


def init_db(path):
    global db
    db = sqlite3.connect(path, check_same_thread=False)
    db.execute("""CREATE TABLE IF NOT EXISTS samples(
        ts REAL, source TEXT, temperature_c REAL, relative_humidity_pct REAL,
        dew_point_c REAL, vpd_kpa REAL, dew_point_margin_c REAL)""")
    db.execute("""CREATE TABLE IF NOT EXISTS forecasts(
        ts REAL, source TEXT, temperature_c REAL, relative_humidity_pct REAL)""")
    db.execute("CREATE INDEX IF NOT EXISTS i_s ON samples(source, ts)")
    db.commit()


def compute_iat(source):
    now = time.time()
    rows = db.execute("SELECT ts,temperature_c,relative_humidity_pct FROM samples "
                      "WHERE source=? AND ts>=? ORDER BY ts", (source, now - watanabe.WINDOW_S)).fetchall()
    h, cov = watanabe.iat_hours(rows, now)
    return {"iat_hours": round(h, 2), "coverage_hours": round(cov, 2),
            "level": watanabe.level(h), "window_complete": cov >= 48.0}


def source_key():
    if state["source"] == "SIMULATED":
        return "simulated"
    if state["source"] == "REPLAY":
        return "replay"
    return "device"


def session_uptime(raw):
    if not isinstance(raw, (int, float)) or not math.isfinite(raw) or raw < 0:
        return None
    last_raw = state["last_raw_uptime"]
    last_session = state["last_session_uptime"]
    if last_raw is not None and raw < last_raw:
        state["uptime_offset"] = (last_session + 60) - raw
    value = raw + state["uptime_offset"]
    state["last_raw_uptime"] = raw
    state["last_session_uptime"] = value
    return value


def build_comparisons():
    actual_by_uptime = {
        sample["session_uptime_s"]: sample
        for sample in state["all_samples"]
        if sample.get("session_uptime_s") is not None
    }
    comparisons = []
    for forecast in state["forecasts"]:
        origin = forecast.get("forecast_uptime_s")
        horizon = forecast.get("horizon_min", 30)
        target = origin + horizon * 60 if origin is not None else None
        actual = actual_by_uptime.get(target)
        comparison = {
            "forecast_uptime_s": origin,
            "target_uptime_s": target,
            "horizon_min": horizon,
            "forecast_temperature_c": forecast.get("temperature_c"),
            "forecast_humidity_pct": forecast.get("relative_humidity_pct"),
            "status": "pending",
        }
        if actual and all(isinstance(forecast.get(key), (int, float))
                          for key in ("temperature_c", "relative_humidity_pct")):
            comparison["status"] = "matched"
            comparison.update({
                "actual_temperature_c": actual["temperature_c"],
                "actual_humidity_pct": actual["relative_humidity_pct"],
                "temperature_error_c": actual["temperature_c"] - forecast["temperature_c"],
                "humidity_error_pct": actual["relative_humidity_pct"] - forecast["relative_humidity_pct"],
            })
        comparisons.append(comparison)
    return list(reversed(comparisons))


def recording_stats(comparisons):
    uptimes = [sample["session_uptime_s"] for sample in state["all_samples"]
               if sample.get("session_uptime_s") is not None]
    return {
        "total_samples": len(state["all_samples"]),
        "total_forecasts": len(state["forecasts"]),
        "duration_seconds": round(max(uptimes) - min(uptimes)) if len(uptimes) > 1 else 0,
        "validated_comparisons": sum(item["status"] == "matched" for item in comparisons),
    }


def ingest(line):
    if not line.startswith(PREFIX):
        return
    try:
        msg = json.loads(line[len(PREFIX):])
    except json.JSONDecodeError:
        return
    with lock:
        msg["received_at"] = msg.pop("_replay_ts", time.time())
        src = source_key()
        if msg.get("type") == "sample":
            msg["session_uptime_s"] = session_uptime(msg.get("uptime_s"))
            state["last_sample_uptime"] = msg["session_uptime_s"]
            state["samples"].append(msg)
            state["all_samples"].append(msg)
            db.execute("INSERT INTO samples VALUES(?,?,?,?,?,?,?)", (
                msg["received_at"], src, msg["temperature_c"], msg["relative_humidity_pct"],
                msg["dew_point_c"], msg["vpd_kpa"], msg["dew_point_margin_c"]))
            db.commit()
        elif msg.get("type") == "forecast":
            msg["forecast_uptime_s"] = state["last_sample_uptime"]
            state["forecast"] = msg
            state["forecasts"].append(msg)
            db.execute("INSERT INTO forecasts VALUES(?,?,?,?)", (
                msg["received_at"], src, msg["temperature_c"], msg["relative_humidity_pct"]))
            db.commit()
        state["last_update"] = msg["received_at"]


def replay(path):
    records = []
    with Path(path).open(encoding="utf-8-sig", errors="replace") as log:
        for line_number, line in enumerate(log, 1):
            text = line.strip()
            if text.startswith(PREFIX):
                text = text[len(PREFIX):]
            elif not text.startswith("{"):
                continue
            try:
                msg = json.loads(text)
            except json.JSONDecodeError as exc:
                print(f"Skipping invalid JSON on line {line_number}: {exc}", file=sys.stderr)
                continue
            if isinstance(msg, dict) and msg.get("type") in ("sample", "forecast"):
                records.append(msg)

    sample_count = sum(msg["type"] == "sample" for msg in records)
    forecast_count = len(records) - sample_count
    if not sample_count:
        raise ValueError("the file contains no @JSON sample records")

    # Preserve uptime spacing so the graph and 48-hour coverage use the
    # recorded timeline even though the file is loaded immediately.
    timeline = []
    last_raw_uptime = None
    last_uptime = None
    uptime_offset = 0.0
    for msg in records:
        if msg["type"] == "sample":
            raw = msg.get("uptime_s")
            if isinstance(raw, (int, float)) and math.isfinite(raw) and raw >= 0:
                if last_raw_uptime is not None and raw < last_raw_uptime:
                    uptime_offset = (last_uptime + 60) - raw
                last_uptime = raw + uptime_offset
                last_raw_uptime = raw
            else:
                last_uptime = (last_uptime + 60) if last_uptime is not None else 0.0
                last_raw_uptime = None
        timeline.append(last_uptime if last_uptime is not None else 0.0)

    time_origin = time.time() - timeline[-1]
    state["source"] = "REPLAY"
    for msg, recorded_uptime in zip(records, timeline):
        msg["_replay_ts"] = time_origin + recorded_uptime
        ingest(PREFIX + json.dumps(msg))
    print(f"Replayed {sample_count} samples and {forecast_count} forecasts from {path}")


def read_serial(port, baud):
    import serial  # pip install pyserial
    state["source"] = "serial:" + port
    with serial.Serial(port, baud, timeout=2) as ser:
        while True:
            ingest(ser.readline().decode("utf-8", "replace").strip())


def simulate(interval):
    state["source"] = "SIMULATED"
    t = 0
    while True:
        temp = 21 + 4 * math.sin(t / 40) + random.uniform(-.2, .2)
        rh = 84 + 9 * math.cos(t / 40) + random.uniform(-.5, .5)
        a = math.log(rh / 100) + 17.27 * temp / (237.7 + temp)
        dew = 237.7 * a / (17.27 - a)
        svp = 0.6108 * math.exp(17.27 * temp / (237.7 + temp))
        risk = ("HIGH" if rh >= 90 and 15 <= temp <= 25 else
                "MODERATE" if rh >= 85 and 10 <= temp <= 30 else "LOW")
        ingest(PREFIX + json.dumps({
            "type": "sample", "uptime_s": t * 60, "temperature_c": round(temp, 2),
            "relative_humidity_pct": round(rh, 2), "dew_point_c": round(dew, 2),
            "vpd_kpa": round(svp * (1 - rh / 100), 3),
            "dew_point_margin_c": round(temp - dew, 2), "risk": risk,
            "risk_basis": "provisional_rule", "forecast_window": min(4, t // 5)}))
        if t % 5 == 0 and t >= 15:
            ingest(PREFIX + json.dumps({
                "type": "forecast", "horizon_min": 30,
                "temperature_c": round(temp + random.uniform(-.5, .5), 2),
                "relative_humidity_pct": round(rh + random.uniform(-2, 2), 2),
                "inference_us": 0}))
        t += 1
        time.sleep(interval)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/api/state":
            with lock:
                comparisons = build_comparisons()
                body = json.dumps({"source": state["source"],
                                   "demo_data": state["demo_data"],
                                   "risk_index": compute_iat(source_key()),
                                   "samples": list(state["samples"]),
                                   "forecast": state["forecast"],
                                   "forecasts": state["forecasts"],
                                   "comparisons": comparisons,
                                   "recording_stats": recording_stats(comparisons),
                                   "last_update": state["last_update"]}).encode()
            ctype = "application/json"
        elif self.path == "/api/export.csv":
            with lock:
                rows = db.execute("SELECT ts,source,temperature_c,relative_humidity_pct,dew_point_c,"
                                  "vpd_kpa,dew_point_margin_c FROM samples ORDER BY ts").fetchall()
            body = ("unix_ts,source,temperature_c,relative_humidity_pct,dew_point_c,vpd_kpa,"
                    "dew_point_margin_c\n" + "\n".join(",".join(map(str, r)) for r in rows)).encode()
            ctype = "text/csv"
        elif self.path in ("/", "/index.html"):
            body = (Path(__file__).parent / "index.html").read_bytes()
            ctype = "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--port", help="serial port, e.g. COM3")
    source.add_argument("--simulate", action="store_true")
    source.add_argument("--replay", metavar="FILE", help="replay @JSON records from a saved serial log")
    ap.add_argument("--demo-data", action="store_true",
                    help="mark replayed synthetic/test data as DEMO DATA")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--sim-interval", type=float, default=1.0)
    ap.add_argument("--http-port", type=int, default=8000)
    ap.add_argument("--db", default=str(DB_PATH), help="SQLite history file")
    a = ap.parse_args()
    if a.demo_data and not a.replay:
        ap.error("--demo-data can only be used with --replay")
    init_db(a.db)
    if a.replay:
        try:
            replay(a.replay)
        except (OSError, ValueError) as exc:
            ap.error(str(exc))
        state["demo_data"] = a.demo_data
        target = None
    elif a.simulate:
        target = lambda: simulate(a.sim_interval)
    else:
        target = lambda: read_serial(a.port, a.baud)
    if target:
        threading.Thread(target=target, daemon=True).start()
    print(f"Dashboard: http://localhost:{a.http_port}")
    ThreadingHTTPServer(("127.0.0.1", a.http_port), Handler).serve_forever()

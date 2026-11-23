"""Reference implementation of the Watanabe et al. (2021) index (see
firmware/watanabe_risk.h for method and assumptions). Used by the dashboard to
recompute the index from stored history, which survives device reboots."""
import math

COEF = [0.00000013611, -0.00001325037, 0.00049243766, -0.00900511846,
        0.083895, -0.3035, 0.436]
CAUTION_H, DANGER_H, WINDOW_S = 15.0, 20.0, 2 * 24 * 3600


def dew_point(t, rh):
    es = 6.1078 * math.exp(17.2693882 * t / (t + 237.3))
    x = math.log(rh * es / 100.0 / 6.1078)
    return 237.3 * x / (17.2693882 - x)


def growth_factor(t):
    if not 3.0 <= t <= 30.0:
        return 0.0
    y = 0.0
    for c in COEF:
        y = y * t + c
    return min(1.0, max(0.0, y))


def weight(t, rh):
    if not 0 < rh <= 100 or t - dew_point(t, rh) >= 1.2:
        return 0.0
    return growth_factor(t)


def level(hours):
    return "DANGER" if hours >= DANGER_H else "CAUTION" if hours >= CAUTION_H else "SAFETY"


def iat_hours(samples, now):
    """samples: iterable of (unix_ts, temp_c, rh), ascending. Each sample is
    weighted by the time to the next sample (capped at 15 min so data gaps are
    not counted as infectable). Returns (hours, covered_hours) over the 2 days
    ending at `now`."""
    pts = [s for s in samples if now - WINDOW_S <= s[0] <= now]
    total = covered = 0.0
    for i, (ts, t, rh) in enumerate(pts):
        nxt = pts[i + 1][0] if i + 1 < len(pts) else min(now, ts + 60)
        dt = min(max(nxt - ts, 0), 900) / 3600.0
        covered += dt
        total += dt * weight(t, rh)
    return total, covered

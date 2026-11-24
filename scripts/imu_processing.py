from __future__ import annotations
import numpy as np
from scipy.signal import savgol_filter
G = 9.81
G_BASE = np.array([0.0, 0.0, -G])

def skew(v) -> np.ndarray:
    (x, y, z) = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])

def resample(t_new, t, y):
    out = np.stack([np.interp(t_new, t, y[:, k]) for k in range(y.shape[1])], axis=1)
    out[(t_new < t[0]) | (t_new > t[-1])] = np.nan
    return out

def imu_to_point(t_imu, gyro_dps, acc_g, calib: dict, r_point_g, fs: float=100.0, window: int=9):
    lag = calib.get('lag_ms', 0.0) / 1000.0
    acc_lag = lag + calib.get('acc_extra_lag_ms', 0.0) / 1000.0
    R = np.asarray(calib['R_gi'])
    bg = np.radians(np.asarray(calib.get('gyro_bias_dps', [0, 0, 0])))
    ba = np.asarray(calib.get('acc_bias_g', [0, 0, 0]))
    sc = np.asarray(calib.get('acc_scale', [1, 1, 1]))
    (t0, t1) = (t_imu[0] - lag, t_imu[-1] - acc_lag)
    grid = np.arange(np.ceil(t0 * fs) / fs, t1, 1.0 / fs)
    gyro = resample(grid + lag, t_imu, np.radians(gyro_dps)) - bg
    acc = (resample(grid + acc_lag, t_imu, acc_g) - ba) / sc * G
    ok = np.all(np.isfinite(gyro), axis=1) & np.all(np.isfinite(acc), axis=1)
    (grid, gyro, acc) = (grid[ok], gyro[ok], acc[ok])
    w = min(window, len(grid) - (1 - len(grid) % 2))
    if w < 5:
        return (grid, gyro @ R.T, np.zeros_like(gyro), acc @ R.T)
    omega = savgol_filter(gyro, w, 3, axis=0) @ R.T
    alpha = savgol_filter(gyro, w, 3, deriv=1, delta=1.0 / fs, axis=0) @ R.T
    r = np.asarray(r_point_g, float)
    f = acc @ R.T + np.cross(alpha, r) + np.cross(omega, np.cross(omega, r))
    return (grid, omega, alpha, f)

def fit_cf_clock(cf_ms: np.ndarray, rx_ms: np.ndarray, bin_ms: float=1000.0, outlier_ms: float=1.0) -> tuple[np.ndarray, dict]:
    cf_ms = np.asarray(cf_ms, float)
    rx_ms = np.asarray(rx_ms, float)
    n = len(cf_ms)
    host = np.full(n, np.nan)
    if n == 0:
        return (host, {'n': 0})
    breaks = np.where(np.diff(cf_ms) < -500)[0] + 1
    seg_stats = []
    for idx in np.split(np.arange(n), breaks):
        (cf, d) = (cf_ms[idx], rx_ms[idx] - cf_ms[idx])
        b = np.floor((cf - cf[0]) / bin_ms).astype(int)
        order = np.lexsort((d, b))
        first = np.r_[True, b[order][1:] != b[order][:-1]]
        sel = order[first]
        (x, y) = (cf[sel] - cf[0], d[sel])
        keep = np.ones(len(sel), bool)
        (a, s) = (float(y.min()), 0.0)
        if len(sel) >= 3:
            for _ in range(4):
                A = np.c_[np.ones(keep.sum()), x[keep]]
                ((a, s), *_) = np.linalg.lstsq(A, y[keep], rcond=None)
                new_keep = y - (a + s * x) < outlier_ms
                if new_keep.sum() < 3 or (new_keep == keep).all():
                    break
                keep = new_keep
        env = a + s * (cf - cf[0])
        host[idx] = cf + env
        env_resid = y[keep] - (a + s * x[keep])
        excess = d - env
        seg_stats.append({'n': int(len(idx)), 'bins': int(len(sel)), 'bins_used': int(keep.sum()), 'drift_ppm': float(s * 1000000.0), 'envelope_jitter_ms': float(np.std(env_resid)) if len(env_resid) > 1 else None, 'rx_excess_median_ms': float(np.median(excess)), 'rx_excess_p95_ms': float(np.percentile(excess, 95))})
    main = max(seg_stats, key=lambda z: z['n'])
    return (host, {**main, 'segments': len(seg_stats)})

"""频域分析：Bode 图、Nyquist 图、稳定裕度。

全部用 numpy 直算，不依赖 scipy.signal.bode（它处理不了延迟）。
"""
from __future__ import annotations

import numpy as np

from .tf import TF, TOL


def log_freq_range(w_min: float = 1e-3, w_max: float = 1e5, n: int = 800) -> np.ndarray:
    """对数等间距频率点。默认覆盖 1e-3 到 1e5 rad/s，够覆盖绝大多数教学对象。"""
    w_min = max(float(w_min), 1e-12)
    w_max = max(float(w_max), w_min * 10)
    return np.logspace(np.log10(w_min), np.log10(w_max), int(n))


def bode(g: TF, w=None, w_min: float = 1e-3, w_max: float = 1e5, n: int = 800) -> dict:
    """返回 {'w','mag_db','phase_deg'}。相位不做 unwrap 跳变处理，保持 -180 附近可读。"""
    if w is None:
        w = log_freq_range(w_min, w_max, n)
    w = np.asarray(w, dtype=float)
    h = g.freqresp(w)
    mag = np.abs(h)
    with np.errstate(divide="ignore"):
        mag_db = 20.0 * np.log10(np.where(mag > TOL, mag, np.nan))
    phase = np.degrees(np.unwrap(np.angle(h)))
    return {"w": w, "mag_db": mag_db, "phase_deg": phase, "h": h}


def _cross(w: np.ndarray, y: np.ndarray, level: float):
    """找 y 穿过 level 的所有位置，返回线性插值后的 x 值列表。"""
    hits = []
    d = y - level
    for i in range(len(d) - 1):
        a, b = d[i], d[i + 1]
        if not np.isfinite(a) or not np.isfinite(b):
            continue
        if a == 0:
            hits.append(w[i])
        elif a * b < 0:
            t = -a / (b - a)
            hits.append(w[i] + t * (w[i + 1] - w[i]))
    return hits


def margins(g: TF, w=None) -> dict:
    """稳定裕度。

    - gain_margin: 相位穿越 -180° 时，增益还能放大多少倍才到 0dB（dB 表示）
    - phase_margin: 增益穿越 0dB 时，相位离 -180° 还差多少度
    返回 nan 表示该指标不存在（曲线没穿越）。
    """
    r = bode(g, w)
    w, mag_db, ph = r["w"], r["mag_db"], r["phase_deg"]

    gm_db = float("nan")
    w_gc = float("nan")
    for wc in _cross(w, ph, -180.0):
        m = float(np.interp(wc, w, mag_db))
        if np.isfinite(m):
            gm_db = -m
            w_gc = wc
            break

    pm_deg = float("nan")
    w_cp = float("nan")
    for w0 in _cross(w, mag_db, 0.0):
        p = float(np.interp(w0, w, ph))
        if np.isfinite(p):
            pm_deg = 180.0 + p
            w_cp = w0
            break

    # 稳定判据：两者都要够。若相位永不到 -180°（gm=inf），只要 PM>0 就算稳。
    gm_ok = (not np.isfinite(gm_db)) or gm_db > 0
    pm_ok = np.isfinite(pm_deg) and pm_deg > 0
    return {
        "gain_margin_db": gm_db,
        "gain_cross_freq": w_gc,
        "phase_margin_deg": pm_deg,
        "phase_cross_freq": w_cp,
        "stable": bool(gm_ok and pm_ok),
    }


def nyquist(g: TF, w=None, w_min: float = 1e-2, w_max: float = 1e3, n: int = 800) -> dict:
    """Nyquist 曲线数据（只算正频率，负频率镜像由前端画）。"""
    if w is None:
        w = log_freq_range(w_min, w_max, n)
    w = np.asarray(w, dtype=float)
    h = g.freqresp(w)
    return {"re": h.real, "im": h.imag, "w": w}


def root_locus(g: TF, k_max: float = 100.0, n: int = 400) -> dict:
    """根轨迹：k 从 0 到 k_max 变化时闭环极点的轨迹。

    闭环特征方程 1 + k·G(s) = 0  =>  den(s) + k·num(s) = 0
    """
    num = g.num
    den = g.den
    ks = np.linspace(0.0, float(k_max), int(n))
    branches = []
    nmax = den.size - 1
    for k in ks:
        poly = np.polyadd(den, k * np.pad(num, (0, max(0, den.size - num.size))))
        p = np.roots(poly) if poly.size > 1 else np.array([])
        p = np.sort_complex(p)
        branches.append(p)
    # 转成 [分支数, k点数]，分支数固定为 den 阶次
    arr = np.full((len(ks), nmax), np.nan + 1j * np.nan)
    for i, p in enumerate(branches):
        m = min(p.size, nmax)
        arr[i, :m] = p[:m]
    # 轨迹连续性：按实部+虚部排序会让分支跳线，这里保持 roots 顺序
    return {"k": ks, "poles": arr, "num": num, "den": den}


def pole_zero_map(g: TF) -> dict:
    """零极点图数据。"""
    z = g.zeros()
    p = g.poles()
    return {"zeros": z, "poles": p}

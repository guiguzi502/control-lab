"""时域仿真：阶跃 / 脉冲 / 斜坡 / 正弦 / 任意输入响应。

数值策略（重要）：
- 控制系统常出现「刚性」问题——闭环极点 -4000 rad/s 配 20s 仿真时长，
  RK4 定步长会爆。所以这里用 **scipy.signal.lsim 的精确矩阵指数解**
  （零阶保持离散化），它对该问题无条件稳定。
- 纯延迟单独处理：仿真后按时间平移重采样，不塞进状态空间。
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from .tf import TF, TOL


def to_state_space(g: TF) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """传递函数 -> (A, B, C, D)，可控标准型。

    G(s) = (b_m s^m + ... + b_0) / (a_n s^n + ... + a_0)
    """
    den = g.den.astype(float)
    num = g.num.astype(float)
    if abs(den[0]) < TOL:
        raise ValueError("分母首项为 0，无法转换")

    den = den / den[0]
    num = num / g.den[0]
    n = den.size - 1
    if n == 0:
        return np.zeros((0, 0)), np.zeros(0), np.zeros(1), float(num[0])

    a = den[1:]
    A = np.zeros((n, n))
    A[0, :] = -a
    if n > 1:
        A[1:, :-1] = np.eye(n - 1)
    B = np.zeros(n)
    B[0] = 1.0

    num_pad = np.concatenate([np.zeros(max(0, n + 1 - num.size)), num])[-(n + 1):]
    D = float(num_pad[0])
    C = (num_pad[1:] - D * a)
    return A, B, C, D


def _apply_delay(t: np.ndarray, y: np.ndarray, delay: float) -> np.ndarray:
    """纯延迟：y(t) -> y(t-T)，线性插值重采样，延迟段补 0。"""
    if abs(delay) < TOL:
        return y
    out = np.zeros_like(y)
    shifted = t - delay
    valid = shifted >= t[0]
    if np.any(valid):
        out[valid] = np.interp(shifted[valid], t, y)
    return out


def suggest_t_end(g: TF, factor: float = 6.0) -> float:
    """按最慢极点推荐仿真时长：t_end ≈ factor / |最慢极点实部|。

    没有极点（纯增益）时返回 10s。
    """
    p = g.poles()
    if p.size == 0:
        return 10.0
    real = np.abs(p.real)
    real = real[real > 1e-9]
    if real.size == 0:
        return 10.0
    slowest = float(np.min(real))
    return float(np.clip(factor / slowest, 0.1, 500.0))


def _auto_dt(g: TF, t_end: float, n_min: int = 2000, n_max: int = 200000) -> float:
    """按最慢/最快极点自动选步长，保证数值稳定且分辨率够。"""
    p = g.poles()
    base = t_end / n_min
    if p.size:
        fastest = float(np.max(np.abs(p.real))) if np.any(p.real) else 0.0
        # 步长要显著小于最快极点的时间常数
        if fastest > 1e-9:
            base = min(base, 0.25 / fastest)
    n = int(np.ceil(t_end / base)) + 1
    n = max(n_min, min(n_max, n))
    return t_end / (n - 1)


def simulate(
    g: TF,
    t_end: float = 10.0,
    dt: float | None = None,
    input_type: str = "step",
    amplitude: float = 1.0,
    t_start: float = 0.0,
    custom_u=None,
    t=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """仿真单输入单输出系统，返回 (t, y, u)。

    dt=None 时按极点自动选步长。
    """
    if t is None:
        if dt is None:
            dt = _auto_dt(g, t_end)
        n = max(2, int(round(t_end / dt)) + 1)
        t = np.linspace(0.0, t_end, n)
    t = np.asarray(t, dtype=float)

    u = np.zeros_like(t)
    x0 = None
    A, B, C, D = to_state_space(g)

    if input_type == "step":
        u[t >= t_start] = amplitude
    elif input_type == "impulse":
        # 单位脉冲响应的精确处理：状态跳变 x(0+) = B·amplitude，之后自由响应。
        # 这比"窄矩形近似"精确得多，且与步长无关。
        if A.shape[0] > 0:
            x0 = B.ravel() * amplitude
        else:
            u[0] = amplitude / max((t[1] - t[0]) if len(t) > 1 else 1e-3, TOL)
    elif input_type == "ramp":
        rel = t - t_start
        u[rel >= 0] = amplitude * rel[rel >= 0]
    elif input_type == "sine":
        rel = t - t_start
        u[rel >= 0] = amplitude * np.sin(rel[rel >= 0])
    elif input_type == "custom":
        if custom_u is None:
            raise ValueError("custom 输入必须提供 custom_u")
        u = np.asarray(custom_u, dtype=float)
        if u.size != t.size:
            u = np.interp(t, np.linspace(t[0], t[-1], u.size), u)
    else:
        raise ValueError(f"未知输入类型: {input_type}")

    if A.shape[0] == 0:
        y = D * u
    else:
        sys = signal.StateSpace(A, B.reshape(-1, 1), C.reshape(1, -1), np.array([[D]]))
        if x0 is None:
            _, y, _ = signal.lsim(sys, U=u, T=t)
        else:
            # 带初始状态的自由响应（脉冲注入）
            _, y, _ = signal.lsim(sys, U=np.zeros_like(t), T=t, X0=x0)

    if abs(g.delay) > TOL:
        y = _apply_delay(t, y, g.delay)
    return t, y, u


def metrics(t: np.ndarray, y: np.ndarray, target: float = 1.0) -> dict:
    """阶跃响应性能指标（超调、调节时间、上升时间、稳态误差）。"""
    out = {
        "final": float("nan"),
        "overshoot": float("nan"),
        "rise_time": float("nan"),
        "settling_time": float("nan"),
        "peak_time": float("nan"),
        "steady_error": float("nan"),
    }
    if t.size < 3:
        return out

    final = float(y[-1])
    out["final"] = final
    if abs(target) > TOL:
        out["steady_error"] = float(target - final)

    if abs(target) < TOL:
        return out

    ipeak = int(np.argmax(y))
    ymax = float(y[ipeak])
    out["peak_time"] = float(t[ipeak])
    # 超调 = (峰值 - 稳态)/目标值
    if np.isfinite(final):
        out["overshoot"] = float(max(0.0, (ymax - final) / abs(target) * 100.0))

    lo, hi = 0.1 * target, 0.9 * target
    i10 = int(np.argmax(y >= lo)) if np.any(y >= lo) else -1
    i90 = int(np.argmax(y >= hi)) if np.any(y >= hi) else -1
    if i10 >= 0 and i90 >= 0 and i90 > i10:
        out["rise_time"] = float(t[i90] - t[i10])

    band = 0.02 * abs(target)
    out_band = np.abs(y - final) > band
    if np.any(out_band):
        last = int(np.flatnonzero(out_band)[-1])
        if last + 1 < t.size:
            out["settling_time"] = float(t[last + 1])
    else:
        out["settling_time"] = float(t[0])
    return out

"""传递函数代数运算。

系数约定与 MATLAB 一致：**降幂排列**。
    s^2 + 3s + 2  ->  [1, 3, 2]

只用 numpy，不依赖 python-control，避免它的版本坑。
"""
from __future__ import annotations

import numpy as np

TOL = 1e-12


def trim(coeffs) -> np.ndarray:
    """去掉首部多余的 0（保证最高次项系数非零）。"""
    a = np.atleast_1d(np.asarray(coeffs, dtype=float)).ravel()
    if a.size == 0:
        return np.array([0.0])
    nz = np.flatnonzero(np.abs(a) > TOL)
    if nz.size == 0:
        return np.array([0.0])
    return a[nz[0]:]


def _safe_atleast(x) -> np.ndarray:
    """把 None / 空 / 标量 都变成合法系数数组。"""
    if x is None:
        return np.array([0.0])
    a = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    if a.size == 0:
        return np.array([0.0])
    return a


def pade(delay: float, order: int = 2):
    """e^{-delay*s} 的 Padé 有理逼近，返回 (num, den) **降幂**系数。

    order=1: (1 - Ts/2) / (1 + Ts/2)      ->  num=[-T/2, 1], den=[T/2, 1]
    order=2: 二阶对称 Padé，直流增益恒为 1
    """
    if abs(delay) < TOL:
        return np.array([1.0]), np.array([1.0])
    n = max(1, int(order))
    from math import factorial

    # 升幂系数：a_k = (2n-k)! n! / ((2n)! k! (n-k)!) · (±T)^k
    asc_num, asc_den = [], []
    for k in range(n + 1):
        base = factorial(2 * n - k) * factorial(n) / (
            factorial(2 * n) * factorial(k) * factorial(n - k)
        )
        asc_num.append(base * ((-delay) ** k))
        asc_den.append(base * (delay ** k))
    # 本项目统一降幂：升幂列表反转即为降幂
    return trim(np.array(asc_num[::-1])), trim(np.array(asc_den[::-1]))


class TF:
    """线性时不变传递函数 G(s) = num(s)/den(s) · e^{-delay·s}。"""

    __slots__ = ("num", "den", "delay")

    def __init__(self, num, den, delay: float = 0.0):
        self.num = trim(_safe_atleast(num))
        self.den = trim(_safe_atleast(den))
        if self.den.size == 1 and abs(self.den[0]) < TOL:
            raise ValueError("分母不能全为 0")
        self.delay = float(delay)

    # ---------- 基本属性 ----------
    @property
    def order(self) -> int:
        return int(self.den.size - 1)

    @property
    def relative_degree(self) -> int:
        return int(self.den.size - self.num.size)

    @property
    def is_proper(self) -> bool:
        """正则（分母阶次 ≥ 分子阶次）。"""
        return self.den.size >= self.num.size

    @property
    def is_strictly_proper(self) -> bool:
        return self.den.size > self.num.size

    def poles(self) -> np.ndarray:
        if self.den.size <= 1:
            return np.array([])
        return np.roots(self.den)

    def zeros(self) -> np.ndarray:
        nz = trim(self.num)
        if nz.size <= 1:
            return np.array([])
        return np.roots(nz)

    def dc_gain(self) -> float:
        """s=0 处的增益（能算才算）。"""
        num0 = self.num[-1] if self.num.size else 0.0
        den0 = self.den[-1] if self.den.size else 1.0
        if abs(den0) < TOL:
            return float("inf")
        return float(num0 / den0)

    def gain(self) -> float:
        """高频增益 lim_{s->inf} G(s)，用于判断正则性。"""
        if self.den.size < self.num.size:
            return float("inf")
        return float(self.num[0] / self.den[0])

    def stability(self) -> str:
        """开环稳定性判断。"""
        p = self.poles()
        if p.size == 0:
            return "无极点（增益环节）"
        real = p.real
        if np.any(real > TOL):
            return "不稳定"
        if np.any(np.abs(real) <= TOL):
            if np.sum(np.abs(real) <= TOL) > 1 or np.any(np.abs(real) <= TOL):
                return "临界稳定"
        return "稳定"

    # ---------- 代数运算 ----------
    def __mul__(self, other) -> "TF":
        """串联。"""
        if isinstance(other, (int, float, complex)):
            return TF(self.num * float(other), self.den, self.delay)
        if not isinstance(other, TF):
            return NotImplemented
        return TF(
            np.polymul(self.num, other.num),
            np.polymul(self.den, other.den),
            self.delay + other.delay,
        )

    __rmul__ = __mul__

    def __add__(self, other) -> "TF":
        """并联。"""
        if isinstance(other, (int, float, complex)):
            other = TF([float(other)], [1.0])
        if not isinstance(other, TF):
            return NotImplemented
        # (n1/d1 + n2/d2) = (n1 d2 + n2 d1) / (d1 d2)
        num = np.polyadd(
            np.polymul(self.num, other.den), np.polymul(other.num, self.den)
        )
        den = np.polymul(self.den, other.den)
        return TF(num, den, max(self.delay, other.delay))

    def __neg__(self) -> "TF":
        return TF(-self.num, self.den, self.delay)

    def __sub__(self, other) -> "TF":
        return self + (-other if isinstance(other, TF) else -float(other))

    def feedback(self, h: "TF | None" = None, sign: int = -1) -> "TF":
        """闭环 Φ = G/(1 + G·H)。

        sign=-1 为负反馈（分母 den·h_den + num·h_num），
        sign=+1 为正反馈（分母 den·h_den - num·h_num）。
        """
        if h is None:
            h = TF([1.0], [1.0])
        elif isinstance(h, (int, float)):
            h = TF([float(h)], [1.0])
        num = np.polymul(self.num, h.den)
        den = np.polyadd(
            np.polymul(self.den, h.den), (-sign) * np.polymul(self.num, h.num)
        )
        return TF(num, den, self.delay)

    def scale(self, k: float) -> "TF":
        return TF(self.num * float(k), self.den, self.delay)

    # ---------- 频域 ----------
    def freqresp(self, w) -> np.ndarray:
        """G(jw) 复数响应；延迟用精确相移 e^{-jwT} 处理。"""
        w = np.asarray(w, dtype=float)
        s = 1j * w
        num = np.polyval(self.num, s)
        den = np.polyval(self.den, s)
        with np.errstate(divide="ignore", invalid="ignore"):
            h = np.where(np.abs(den) > TOL, num / den, np.nan + 1j * np.nan)
        if abs(self.delay) > TOL:
            h = h * np.exp(-s * self.delay)
        return h

    # ---------- 展示 ----------
    def describe(self) -> str:
        def poly_str(c) -> str:
            c = trim(c)
            n = c.size - 1
            parts = []
            for i, v in enumerate(c):
                p = n - i
                if abs(v) < TOL:
                    continue
                vv = abs(v)
                coef = "" if abs(vv - 1) < 1e-9 else f"{vv:g}"
                if p == 0:
                    term = f"{coef}"
                elif p == 1:
                    term = f"{coef}s"
                else:
                    term = f"{coef}s^{p}"
                parts.append(term if v > 0 else f"- {term}")
            if not parts:
                return "0"
            out = parts[0]
            for t in parts[1:]:
                out += f" {t}" if t.startswith("-") else f" + {t}"
            return out

        s = f"({poly_str(self.num)}) / ({poly_str(self.den)})"
        if abs(self.delay) > TOL:
            s += f" · e^(-{self.delay:g}s)"
        return s

    def __repr__(self) -> str:  # pragma: no cover
        return f"TF[{self.describe()}]"


def tf_from_roots(z, p, gain: float = 1.0) -> TF:
    """零极点形式转多项式：G(s) = k·Π(s-z_i)/Π(s-p_j)。"""
    num = np.poly(np.asarray(z, dtype=complex)) if len(z) else np.array([1.0])
    den = np.poly(np.asarray(p, dtype=complex)) if len(p) else np.array([1.0])
    return TF(np.real(num * gain), np.real(den))


def pid(kp: float = 1.0, ki: float = 0.0, kd: float = 0.0, n_filter: float = 0.0) -> TF:
    """PID 控制器。

    C(s) = kp + ki/s + kd·s/(1 + s/n_filter)
    n_filter=0 表示理想微分（不推荐用于仿真，会放噪声）。
    """
    C = TF([0.0], [1.0])  # 零传递函数
    if abs(kp) > TOL:
        C = C + TF([kp], [1.0])
    if abs(ki) > TOL:
        C = C + TF([ki], [1.0, 0.0])
    if abs(kd) > TOL:
        if n_filter > TOL:
            # kd·s/(1+s/N) = kd·N·s/(s+N)
            C = C + TF([kd * n_filter, 0.0], [1.0, n_filter])
        else:
            C = C + TF([kd, 0.0], [1.0])
    if not np.any(np.abs(C.num) > TOL):
        return TF([0.0], [1.0])
    return C

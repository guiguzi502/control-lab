"""ctrlkit 单元测试。

原则：用解析解验证数值结果，不自己验证自己。
"""
import numpy as np
import pytest

from ctrlkit import (
    TF,
    pid,
    tf_from_roots,
    pade,
    simulate,
    metrics,
    bode,
    margins,
    nyquist,
    root_locus,
    get_preset,
    list_presets,
)


class TestTFBasics:
    def test_trim_leading_zeros(self):
        g = TF([0, 0, 1.0], [0, 1.0, 2.0])
        assert np.allclose(g.num, [1.0])
        assert np.allclose(g.den, [1.0, 2.0])

    def test_order(self):
        assert TF([1.0], [1.0, 2.0, 3.0]).order == 2

    def test_dc_gain(self):
        # 1/(s+2) 在 s=0 处 = 0.5
        assert TF([1.0], [1.0, 2.0]).dc_gain() == pytest.approx(0.5)

    def test_poles_zeros(self):
        g = tf_from_roots([-3.0], [-1.0, -2.0])
        assert np.allclose(sorted(g.poles().real), [-2.0, -1.0])
        assert np.allclose(g.zeros().real, [-3.0])

    def test_stability(self):
        assert TF([1.0], [1.0, 1.0]).stability() == "稳定"
        assert TF([1.0], [1.0, -1.0]).stability() == "不稳定"


class TestTFAlgebra:
    def test_series(self):
        g1 = TF([1.0], [1.0, 1.0])   # 1/(s+1)
        g2 = TF([1.0], [1.0, 2.0])   # 1/(s+2)
        g = g1 * g2
        assert np.allclose(g.den, [1.0, 3.0, 2.0])

    def test_parallel(self):
        g1 = TF([1.0], [1.0, 1.0])
        g2 = TF([1.0], [1.0, 2.0])
        g = g1 + g2
        # 1/(s+1) + 1/(s+2) = (2s+3)/(s^2+3s+2)
        assert np.allclose(g.num, [2.0, 3.0])
        assert np.allclose(g.den, [1.0, 3.0, 2.0])

    def test_feedback_unity(self):
        # G=1/(s+1)，负反馈后 Φ=1/(s+2)
        g = TF([1.0], [1.0, 1.0])
        phi = g.feedback()
        assert np.allclose(phi.num, [1.0])
        assert np.allclose(phi.den, [1.0, 2.0])

    def test_feedback_with_h(self):
        # Φ = G/(1+GH)，H=2，G=1/(s+1) => 1/(s+3)
        g = TF([1.0], [1.0, 1.0])
        phi = g.feedback(TF([2.0], [1.0]))
        assert np.allclose(phi.den, [1.0, 3.0])


class TestPID:
    def test_p_only(self):
        c = pid(kp=2.0)
        assert np.allclose(c.num, [2.0])
        assert np.allclose(c.den, [1.0])

    def test_pi(self):
        c = pid(kp=1.0, ki=2.0)
        # 1 + 2/s = (s+2)/s
        assert np.allclose(c.num, [1.0, 2.0])
        assert np.allclose(c.den, [1.0, 0.0])

    def test_pid_with_filter(self):
        c = pid(kp=1.0, ki=0.0, kd=1.0, n_filter=10.0)
        # 1 + 10s/(s+10) = (11s+10)/(s+10)
        assert np.allclose(c.num / c.num[0], [1.0, 10.0 / 11.0])

    def test_all_zero(self):
        c = pid(0.0, 0.0, 0.0)
        assert c.num.size == 1


class TestPade:
    def test_zero_delay(self):
        n, d = pade(0.0)
        assert np.allclose(n, [1.0]) and np.allclose(d, [1.0])

    def test_first_order_delay_approx(self):
        # 一阶 Padé: (1-Ts/2)/(1+Ts/2)，T=1 -> (1-0.5s)/(1+0.5s)
        # 降幂系数：num = -0.5s + 1 -> [-0.5, 1]；den = 0.5s + 1 -> [0.5, 1]
        n, d = pade(1.0, order=1)
        assert np.allclose(n, [-0.5, 1.0])
        assert np.allclose(d, [0.5, 1.0])

    def test_dc_gain_approx_one(self):
        n, d = pade(0.7, order=2)
        assert (n[-1] / d[-1]) / (n[0] / d[0]) == pytest.approx(1.0, abs=1e-9)


class TestSimulation:
    def test_first_order_step_final_value(self):
        # 1/(s+1) 阶跃响应终值 = 1
        g = TF([1.0], [1.0, 1.0])
        t, y, u = simulate(g, t_end=10.0, dt=0.01)
        assert y[-1] == pytest.approx(1.0, abs=1e-3)
        assert np.allclose(u, 1.0)

    def test_first_order_time_constant(self):
        # 一阶系统在 t=T 时达到 63.2%
        g = TF([1.0], [1.0, 1.0])
        t, y, u = simulate(g, t_end=5.0, dt=0.001)
        idx = np.argmin(np.abs(t - 1.0))
        assert y[idx] == pytest.approx(0.632, abs=5e-3)

    def test_second_order_overshoot_matches_theory(self):
        # 标准二阶：zeta=0.5 -> 超调 = exp(-pi*zeta/sqrt(1-zeta^2)) = 16.3%
        zeta, wn = 0.5, 2.0
        num = [wn ** 2]
        den = [1.0, 2 * zeta * wn, wn ** 2]
        g = TF(num, den)
        t, y, u = simulate(g, t_end=15.0, dt=0.001)
        m = metrics(t, y, target=1.0)
        assert m["overshoot"] == pytest.approx(16.3, abs=1.5)

    def test_impulse_response_area(self):
        # 脉冲响应 1/(s+1) 的积分应趋近 1
        g = TF([1.0], [1.0, 1.0])
        t, y, u = simulate(g, t_end=20.0, dt=0.001, input_type="impulse")
        area = np.trapezoid(y, t)
        assert area == pytest.approx(1.0, abs=0.05)

    def test_delay_shifts_response(self):
        # 带 1s 延迟，y 在 t<1 时应为 0
        g = TF([1.0], [1.0, 1.0], delay=1.0)
        t, y, u = simulate(g, t_end=5.0, dt=0.01)
        mask = t < 0.95
        assert np.allclose(y[mask], 0.0, atol=1e-9)

    def test_zero_order_gain(self):
        # 纯增益 2
        g = TF([2.0], [1.0])
        t, y, u = simulate(g, t_end=1.0, dt=0.1)
        assert np.allclose(y, 2.0)

    def test_unstable_diverges(self):
        # 1/(s-1) 阶跃发散
        g = TF([1.0], [1.0, -1.0])
        t, y, u = simulate(g, t_end=3.0, dt=0.001)
        assert y[-1] > 10.0


class TestMetrics:
    def test_steady_state_error(self):
        t = np.linspace(0, 10, 1000)
        y = np.full_like(t, 0.8)
        m = metrics(t, y, target=1.0)
        assert m["steady_error"] == pytest.approx(0.2, abs=1e-6)


class TestBode:
    def test_integrator_slope(self):
        # 1/s 的幅频斜率 -20dB/dec
        g = TF([1.0], [1.0, 0.0])
        r = bode(g, w=np.array([1.0, 10.0]))
        slope = r["mag_db"][1] - r["mag_db"][0]
        assert slope == pytest.approx(-20.0, abs=0.5)

    def test_lowpass_corner(self):
        # 1/(s+1) 在 w=1 处 -3dB
        g = TF([1.0], [1.0, 1.0])
        r = bode(g, w=np.array([1.0]))
        assert r["mag_db"][0] == pytest.approx(-3.01, abs=0.1)

    def test_phase_first_order(self):
        # 1/(s+1) 在 w=1 处相位 -45°
        g = TF([1.0], [1.0, 1.0])
        r = bode(g, w=np.array([1.0]))
        assert r["phase_deg"][0] == pytest.approx(-45.0, abs=1.0)

    def test_delay_phase_shift(self):
        # 纯增益+延迟：相位 = -wT
        g = TF([1.0], [1.0], delay=0.5)
        r = bode(g, w=np.array([2.0]))
        assert r["phase_deg"][0] == pytest.approx(-57.3, abs=1.5)


class TestMargins:
    def test_integrator_pid_margins(self):
        # G=1/s，Kp=1，增益裕度无穷，相位裕度 90°
        g = TF([1.0], [1.0, 0.0])
        m = margins(g)
        assert m["phase_margin_deg"] == pytest.approx(90.0, abs=2.0)

    def test_stable_system(self):
        # 1/(s²+3s+2) 直流增益 0.5，幅值始终 < 0dB，不存在增益穿越 -> PM 为 nan
        g = TF([1.0], [1.0, 3.0, 2.0])
        m = margins(g)
        assert not np.isfinite(m["phase_margin_deg"])

    def test_gain_crossing_system(self):
        # 10/(s(s+1)) 有 0dB 穿越，PM 应为正
        g = TF([10.0], [1.0, 1.0, 0.0])
        m = margins(g)
        assert np.isfinite(m["phase_margin_deg"])
        assert m["phase_margin_deg"] > 0


class TestRootLocus:
    def test_shape(self):
        g = TF([1.0], [1.0, 2.0, 1.0])
        r = root_locus(g, k_max=10.0, n=50)
        assert r["poles"].shape == (50, 2)

    def test_start_at_open_loop_poles(self):
        g = TF([1.0], [1.0, 3.0, 2.0])
        r = root_locus(g, k_max=10.0, n=20)
        start = r["poles"][0]
        expect = np.sort_complex(g.poles())
        assert np.allclose(np.sort_complex(start), expect, atol=1e-6)


class TestNyquist:
    def test_returns_complex(self):
        g = TF([1.0], [1.0, 1.0])
        r = nyquist(g, n=100)
        assert len(r["re"]) == 100 and len(r["im"]) == 100


class TestPresets:
    def test_all_loadable(self):
        for item in list_presets():
            g = get_preset(item["key"])
            assert isinstance(g, TF)

    def test_unknown_raises(self):
        with pytest.raises(KeyError):
            get_preset("nonexistent")

    def test_stability_of_presets(self):
        assert get_preset("first_order").stability() == "稳定"
        assert get_preset("unstable").stability() == "不稳定"

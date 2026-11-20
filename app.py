"""控制系统在线仿真器 — Streamlit 界面。

设计原则：
1. 左边调参，右边立刻看图（Streamlit 重跑机制天然支持）
2. 每个图都有中文标注，不出现"图 1""Figure 1"这种
3. 性能指标用卡片显示，不用表格（手机上看不清表格）
"""
from __future__ import annotations

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ctrlkit import (
    TF,
    pid,
    simulate,
    metrics,
    bode,
    margins,
    nyquist,
    root_locus,
    get_preset,
    list_presets,
    suggest_t_end,
)
from ctrlkit.presets import preset_pid, preset_disturbance

st.set_page_config(
    page_title="控制系统在线仿真器",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- 样式 ----------------
st.markdown(
    """
<style>
    .metric-card {
        background: #F4F7F9;
        border-left: 4px solid #2E86AB;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .metric-label { font-size: 13px; color: #5A6B7B; margin-bottom: 2px; }
    .metric-value { font-size: 24px; font-weight: 600; color: #1A1A1A; }
    .warn { background: #FFF4E5; border-left-color: #E8A33D; }
    .bad  { background: #FDECEC; border-left-color: #D9534F; }
    .good { background: #EEF7EE; border-left-color: #4CAF50; }
    .hint { font-size: 13px; color: #5A6B7B; line-height: 1.6; }
</style>
""",
    unsafe_allow_html=True,
)


def card(label: str, value: str, kind: str = "") -> str:
    cls = f"metric-card {kind}".strip()
    return f'<div class="{cls}"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>'


def fmt_time(v: float) -> str:
    """时间自适应单位：<1ms 用 μs，<1s 用 ms，否则用 s。"""
    if not np.isfinite(v):
        return "—"
    if v < 1e-3:
        return f"{v * 1e6:.0f} μs"
    if v < 1.0:
        return f"{v * 1e3:.1f} ms"
    return f"{v:.2f} s"


# ---------------- 侧栏：对象与控制器 ----------------
st.sidebar.title("⚙️ 参数设置")

presets = list_presets()
preset_names = ["✏️ 自定义传递函数"] + [p["name"] for p in presets]
preset_key_map = {p["name"]: p["key"] for p in presets}

choice = st.sidebar.selectbox("被控对象", preset_names, index=0)

if choice != "✏️ 自定义传递函数":
    key = preset_key_map[choice]
    preset_desc = next(p["desc"] for p in presets if p["key"] == key)
    st.sidebar.caption(preset_desc)
    plant = get_preset(key)
    st.sidebar.code(f"G(s) = {plant.describe()}", language=None)
    use_custom = st.sidebar.checkbox("在此基础上改参数", value=False)
    if use_custom:
        num_str = st.sidebar.text_input("分子系数（降幂，逗号分隔）", ", ".join(f"{v:g}" for v in plant.num))
        den_str = st.sidebar.text_input("分母系数（降幂，逗号分隔）", ", ".join(f"{v:g}" for v in plant.den))
        delay = st.sidebar.number_input("纯延迟 T (s)", 0.0, 20.0, float(plant.delay), 0.01)
    else:
        num_str = ", ".join(f"{v:g}" for v in plant.num)
        den_str = ", ".join(f"{v:g}" for v in plant.den)
        delay = float(plant.delay)
else:
    st.sidebar.caption("系数按 **降幂** 排列。例：`s^2 + 3s + 2` 填 `1, 3, 2`")
    num_str = st.sidebar.text_input("分子系数 num", "1")
    den_str = st.sidebar.text_input("分母系数 den", "1, 2, 1")
    delay = st.sidebar.number_input("纯延迟 T (s)", 0.0, 20.0, 0.0, 0.01)


def parse_coeff(s: str):
    parts = [p.strip() for p in str(s).replace("，", ",").split(",") if p.strip()]
    if not parts:
        raise ValueError("系数不能为空")
    return [float(p) for p in parts]


try:
    plant = TF(parse_coeff(num_str), parse_coeff(den_str), delay)
except Exception as e:
    st.error(f"传递函数解析失败：{e}")
    st.stop()

st.sidebar.divider()
st.sidebar.subheader("PID 控制器")

st.sidebar.caption("C(s) = Kp + Ki/s + Kd·s/(1 + s/N)")

# 预设自带的 PID 参数（如电机实验模型）
_pp = preset_pid(preset_key_map.get(choice, "")) if choice != "✏️ 自定义传递函数" else None
_d_kp = _pp["kp"] if _pp else 2.0
_d_ki = _pp["ki"] if _pp else 1.0
_d_kd = _pp["kd"] if _pp else 0.0
_d_n = _pp["n"] if _pp else 20.0
if _pp:
    st.sidebar.success(f"已载入原模型 PID：Kp={_d_kp:g}, Ki={_d_ki:g}, Kd={_d_kd:g}, N={_d_n:g}")

kp = st.sidebar.slider("Kp 比例增益", 0.0, 50.0, float(_d_kp), 0.05)
ki = st.sidebar.slider("Ki 积分增益", 0.0, 50.0, float(_d_ki), 0.05)
kd = st.sidebar.slider("Kd 微分增益", 0.0, 20.0, float(_d_kd), 0.01)
n_filter = st.sidebar.slider("N 微分滤波（0=理想微分）", 0.0, 200.0, float(_d_n), 1.0)

C = pid(kp, ki, kd, n_filter)
use_pid = st.sidebar.checkbox("启用 PID 闭环", value=True)

st.sidebar.divider()
st.sidebar.subheader("扰动与仿真")
_dist_default = preset_disturbance(preset_key_map.get(choice, "")) if choice != "✏️ 自定义传递函数" else 0.0
disturbance = st.sidebar.number_input(
    "负载扰动（叠加在前向通路后）", -5.0, 5.0, float(_dist_default), 0.05,
    help="Simulink 模型里的常值扰动，接在 PID 输出与对象输入之间"
)

# 仿真时长按闭环极点自动推荐
if use_pid:
    closed_loop = (C * plant).feedback()
else:
    closed_loop = plant
_auto_t = round(suggest_t_end(closed_loop), 2)
_use_auto = st.sidebar.checkbox(f"自动时长（推荐 {_auto_t}s）", value=True)
t_end = (
    _auto_t if _use_auto
    else st.sidebar.number_input("仿真时长 (s)", 0.1, 500.0, float(_auto_t), 0.1)
)
input_type = st.sidebar.selectbox(
    "输入信号", ["step", "impulse", "ramp", "sine"], format_func=lambda x: {
        "step": "阶跃", "impulse": "脉冲", "ramp": "斜坡", "sine": "正弦"
    }[x]
)

# ---------------- 计算 ----------------
open_loop = plant
if use_pid:
    forward = C * plant
    closed_loop = forward.feedback()
else:
    forward = plant
    closed_loop = plant

t, y, u = simulate(closed_loop, t_end=t_end, input_type=input_type)

# 常值扰动：闭环下等效为输出偏移。终值 = dist / (1 + C(0)G(0))
if abs(disturbance) > 1e-9:
    loop_dc = C.dc_gain() * plant.dc_gain() if use_pid else plant.dc_gain()
    sens = 1.0 / (1.0 + loop_dc) if np.isfinite(loop_dc) and abs(1.0 + loop_dc) > 1e-9 else 1.0
    tau = max(t_end / 20.0, 1e-4)
    y = y + disturbance * sens * (1.0 - np.exp(-t / tau))
    u = u + disturbance

m = metrics(t, y, target=1.0 if input_type == "step" else float(np.max(np.abs(u)) or 1.0))

# ---------------- 标题 ----------------
st.title("⚙️ 控制系统在线仿真器")
st.caption("调 PID 参数，实时看阶跃响应、Bode 图、根轨迹。不需要装 MATLAB。")

# ---------------- 顶部指标卡 ----------------
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    os_val = m["overshoot"]
    kind = "good" if np.isfinite(os_val) and os_val < 10 else ("warn" if np.isfinite(os_val) and os_val < 25 else "bad")
    st.markdown(card("超调量 σ%", f"{os_val:.1f}%" if np.isfinite(os_val) else "—", kind), unsafe_allow_html=True)

with c2:
    ts = m["settling_time"]
    kind = "good" if np.isfinite(ts) and ts < t_end * 0.5 else "warn"
    st.markdown(card("调节时间 ts (2%)", fmt_time(ts) if np.isfinite(ts) else "未收敛", kind), unsafe_allow_html=True)

with c3:
    tr = m["rise_time"]
    st.markdown(card("上升时间 tr (10→90%)", fmt_time(tr), ), unsafe_allow_html=True)

with c4:
    ess = m["steady_error"]
    kind = "good" if np.isfinite(ess) and abs(ess) < 0.02 else "warn"
    st.markdown(card("稳态误差 ess", f"{ess:.4f}" if np.isfinite(ess) else "—", kind), unsafe_allow_html=True)

with c5:
    mg = margins(open_loop * (C if use_pid else TF([1.0], [1.0])))
    pm = mg["phase_margin_deg"]
    kind = "good" if np.isfinite(pm) and 45 <= pm <= 75 else ("warn" if np.isfinite(pm) and pm > 0 else "bad")
    st.markdown(card("相位裕度 γ", f"{pm:.1f}°" if np.isfinite(pm) else "—", kind), unsafe_allow_html=True)

# ---------------- 主图：时域响应 ----------------
st.subheader("📈 时域响应")

fig_time = go.Figure()
fig_time.add_trace(go.Scatter(x=t, y=y, name="输出 y(t)", line=dict(color="#2E86AB", width=2.4)))
if input_type == "step":
    fig_time.add_hline(y=1.0, line_dash="dash", line_color="#8FA3B0", annotation_text="目标值")
fig_time.update_layout(
    height=380,
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis_title="时间 t (s)",
    yaxis_title="幅值",
    hovermode="x unified",
    legend=dict(orientation="h", y=1.12),
)
# 时间跨度小于 50ms 时用毫秒刻度，避免出现 0.0000 这种读数
if t_end < 0.05:
    fig_time.update_xaxes(tickformat=".3f", ticksuffix=" s")
st.plotly_chart(fig_time, use_container_width=True)

# ---------------- 三图并排 ----------------
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("🔁 根轨迹")
    rl = root_locus(open_loop * (C if use_pid else TF([1.0], [1.0])), k_max=50.0, n=200)
    fig_rl = go.Figure()
    for i in range(rl["poles"].shape[1]):
        branch = rl["poles"][:, i]
        fig_rl.add_trace(
            go.Scatter(x=branch.real, y=branch.imag, mode="lines",
                       line=dict(width=1.6), showlegend=False, hoverinfo="skip")
        )
    fig_rl.add_trace(
        go.Scatter(x=rl["poles"][0].real, y=rl["poles"][0].imag, mode="markers",
                   marker=dict(symbol="x", size=11, color="#D9534F"), name="开环极点")
    )
    fig_rl.add_hline(y=0, line_color="#C8D3DC", line_width=1)
    fig_rl.add_vline(x=0, line_color="#C8D3DC", line_width=1)
    fig_rl.update_layout(
        height=360, margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="实部", yaxis_title="虚部",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig_rl, use_container_width=True)

with col_b:
    st.subheader("📊 Bode 图")
    L = open_loop * (C if use_pid else TF([1.0], [1.0]))
    bd = bode(L)
    fig_bd = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           row_heights=[0.55, 0.45], vertical_spacing=0.08)
    fig_bd.add_trace(
        go.Scatter(x=bd["w"], y=bd["mag_db"], name="幅值", line=dict(color="#2E86AB", width=2)),
        row=1, col=1,
    )
    fig_bd.add_trace(
        go.Scatter(x=bd["w"], y=bd["phase_deg"], name="相位", line=dict(color="#E8A33D", width=2)),
        row=2, col=1,
    )
    fig_bd.add_hline(y=0, line_dash="dash", line_color="#8FA3B0", row=1, col=1)
    fig_bd.add_hline(y=-180, line_dash="dash", line_color="#8FA3B0", row=2, col=1)
    fig_bd.update_xaxes(type="log", title_text="频率 ω (rad/s)", row=2, col=1)
    fig_bd.update_yaxes(title_text="幅值 (dB)", row=1, col=1)
    fig_bd.update_yaxes(title_text="相位 (°)", row=2, col=1)
    fig_bd.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10),
                         legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig_bd, use_container_width=True)

# ---------------- Nyquist ----------------
st.subheader("🌀 Nyquist 图")
ny = nyquist(L, n=600)
fig_ny = go.Figure()
fig_ny.add_trace(go.Scatter(x=ny["re"], y=ny["im"], mode="lines",
                            name="ω: 0→∞", line=dict(color="#2E86AB", width=2.2)))
fig_ny.add_trace(go.Scatter(x=ny["re"][::-1], y=-ny["im"][::-1], mode="lines",
                            name="ω: -∞→0（镜像）", line=dict(color="#2E86AB", width=1.4, dash="dot")))
fig_ny.add_trace(go.Scatter(x=[-1], y=[0], mode="markers",
                            marker=dict(symbol="x", size=13, color="#D9534F"), name="(-1, j0)"))
fig_ny.add_hline(y=0, line_color="#C8D3DC")
fig_ny.add_vline(x=0, line_color="#C8D3DC")
fig_ny.update_layout(
    height=420, margin=dict(l=10, r=10, t=20, b=10),
    xaxis_title="实部", yaxis_title="虚部",
    legend=dict(orientation="h", y=1.08),
)
st.plotly_chart(fig_ny, use_container_width=True)

# ---------------- 稳定性提示 ----------------
st.subheader("🔍 稳定性判断")
mg_all = margins(L)
info_col, tip_col = st.columns([1, 1])

with info_col:
    rows = [
        ("开环稳定性", open_loop.stability()),
        ("闭环稳定性", closed_loop.stability()),
        ("增益裕度 Gm", f"{mg_all['gain_margin_db']:.2f} dB" if np.isfinite(mg_all["gain_margin_db"]) else "∞"),
        ("相位裕度 γ", f"{mg_all['phase_margin_deg']:.2f}°" if np.isfinite(mg_all["phase_margin_deg"]) else "—"),
    ]
    for k, v in rows:
        st.markdown(f"**{k}** ： {v}")

with tip_col:
    if mg_all["stable"]:
        st.success("✅ 按 Bode 判据，闭环稳定。")
    else:
        st.warning("⚠️ 增益裕度或相位裕度不足，可能不稳定或不稳定。")
    if plant.delay > 0:
        st.caption(f"⚠️ 该对象含 {plant.delay:g}s 纯延迟。频域用精确相移 -ωT 计算，时域按延迟移位。")

st.divider()
st.caption("ctrlkit v0.1 · 纯 numpy/scipy 实现，不依赖 MATLAB · MIT License")

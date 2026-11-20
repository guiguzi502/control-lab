"""预设被控对象（教学常用模型）。"""
from __future__ import annotations

from .tf import TF

PRESETS = {
    "dc_motor": {
        "name": "直流电机（二阶）",
        "desc": "电枢控制直流电机，忽略电感时近似为一阶；含电感为二阶。典型工业对象。",
        "num": [1.0],
        "den": [1.0, 2.0, 1.0],
        "delay": 0.0,
    },
    "double_integrator": {
        "name": "双积分器（1/s²）",
        "desc": "位置控制系统的最简模型，如卫星姿态、小车位移。临界稳定，必须闭环。",
        "num": [1.0],
        "den": [1.0, 0.0, 0.0],
        "delay": 0.0,
    },
    "first_order": {
        "name": "一阶惯性（1/(Ts+1)）",
        "desc": "温度、液位、RC 电路。T=1s，无超调，一阶系统标配。",
        "num": [1.0],
        "den": [1.0, 1.0],
        "delay": 0.0,
    },
    "second_order_underdamped": {
        "name": "二阶欠阻尼（ζ=0.3, ωn=2）",
        "desc": "典型振荡环节，超调约 37%，用来观察阻尼比的影响最直观。",
        "num": [4.0],
        "den": [1.0, 1.2, 4.0],
        "delay": 0.0,
    },
    "third_order": {
        "name": "三阶系统（含一个快极点）",
        "desc": "1/((s+1)(s+2)(s+5))，讨论主导极点近似时用。",
        "num": [10.0],
        "den": [1.0, 8.0, 17.0, 10.0],
        "delay": 0.0,
    },
    "delay_process": {
        "name": "带纯延迟的一阶（FOPDT）",
        "desc": "1/(s+1)·e^{-0.5s}。过程工业最常见：传输延迟、采样延迟。",
        "num": [1.0],
        "den": [1.0, 1.0],
        "delay": 0.5,
    },
    "unstable": {
        "name": "不稳定对象（1/(s-1)）",
        "desc": "开环不稳定，必须反馈镇定。用来演示为什么 PID 有时救不回来。",
        "num": [1.0],
        "den": [1.0, -1.0],
        "delay": 0.0,
    },
    "integrator_delay": {
        "name": "积分+延迟（e^{-s}/s）",
        "desc": "积分器串联纯延迟，最难整定的对象之一，延迟超过临界值就无法用 P 控制稳定。",
        "num": [1.0],
        "den": [1.0, 0.0],
        "delay": 1.0,
    },
    # ---- 来自实测/课程实验的真实模型 ----
    "motor_pid_lab": {
        "name": "⭐ 直流电机调速（课程实验原模型）",
        "desc": "G(s)=50/(0.1s+1)，串并联 PID：Kp=3, Ki=15, Kd=0.05, N=100，叠加 0.3 负载扰动。"
                "这是 Simulink「motor_PID.slx」同款结构与参数，可直接对照你当年的实验结果。",
        "num": [50.0],
        "den": [0.1, 1.0],
        "delay": 0.0,
        "pid": {"kp": 3.0, "ki": 15.0, "kd": 0.05, "n": 100.0},
        "disturbance": 0.3,
    },
}


def get_preset(key: str) -> TF:
    """按 key 取预设传递函数。"""
    if key not in PRESETS:
        raise KeyError(f"未知预设: {key}，可用: {list(PRESETS)}")
    p = PRESETS[key]
    return TF(p["num"], p["den"], p["delay"])


def list_presets() -> list:
    """列出所有预设（name/desc/key），供前端下拉框用。"""
    return [
        {"key": k, "name": v["name"], "desc": v["desc"]}
        for k, v in PRESETS.items()
    ]


def preset_pid(key: str) -> dict:
    """取预设自带的 PID 参数；没有则返回 None。"""
    p = PRESETS.get(key, {})
    return p.get("pid")


def preset_disturbance(key: str) -> float:
    """取预设自带的常值扰动；没有则返回 0。"""
    return float(PRESETS.get(key, {}).get("disturbance", 0.0))

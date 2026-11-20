"""ctrlkit — 轻量控制系统工具箱。

不依赖 MATLAB、不依赖 python-control，纯 numpy/scipy。
面向教学与在线仿真：整定 PID、看阶跃响应、画 Bode/Nyquist/根轨迹。
"""
from .tf import TF, pid, tf_from_roots, pade
from .sim import simulate, metrics, to_state_space, suggest_t_end
from .freq import bode, margins, nyquist, root_locus, pole_zero_map, log_freq_range
from .presets import PRESETS, get_preset, list_presets, preset_pid, preset_disturbance

__version__ = "0.1.0"
__all__ = [
    "TF",
    "pid",
    "tf_from_roots",
    "pade",
    "simulate",
    "metrics",
    "to_state_space",
    "suggest_t_end",
    "bode",
    "margins",
    "nyquist",
    "root_locus",
    "pole_zero_map",
    "log_freq_range",
    "PRESETS",
    "get_preset",
    "list_presets",
    "preset_pid",
    "preset_disturbance",
]

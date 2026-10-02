"""演示信号合成：让整个实验台不依赖任何外部音频文件。

为什么合成而不是附带 wav 文件：
1. 可重现 —— git 里一个二进制 wav 改没改过没人知道，
   合成参数就是音频的「源代码」；
2. 可断言 —— 合成 /a/ 时基频就是 120Hz，测试可以直接断言
   基频追踪输出 120±5Hz，这是「检测算法正确」的最硬证据；
3. 可控 —— 想演示 VAD 就合成「语音段+静音段」，
   想演示过零率就合成浊音 vs 摩擦音，信号即考题。
"""

from __future__ import annotations

import math
import random

from .wavio import write_wav


def _fade(edges: list[float], total: int, fade_n: int = 100) -> None:
    """首尾各 fade_n 个样本线性淡入淡出，防止波形跳变产生咔哒声。"""
    for i in range(min(fade_n, len(edges))):
        g = i / fade_n
        edges[i] *= g
        edges[-1 - i] *= g


def synth_vowel(path: str, sample_rate: int = 16000, duration: float = 1.0,
                f0: float = 120.0, formants: tuple[float, ...] = (730, 1090, 2440),
                jitter: float = 0.005, seed: int = 42) -> dict:
    """合成元音 /a/：基频脉冲串 × 共振峰包络。

    声源-滤波器模型（source-filter）：
    - 声源：准周期的声门脉冲，这里用锯齿谐波和近似，加 0.5% 抖动
      （真实基频从来不是分毫不差，抖动让自相关不至于「太容易」）；
    - 滤波器：F1/F2/F3 三个共振峰，每个是二阶谐振峰，
      在谐波谱上表现为被放大到 Q 倍的频带。

    /a/ 的共振峰取 (730, 1090, 2440)Hz，这是 Peterson & Barney (1952)
    表里的经典男声数值 —— 教学上顺带讲一句「共振峰=元音身份」。
    """
    rng = random.Random(seed)
    n = int(sample_rate * duration)
    out = _resonator_synth(n, sample_rate, f0, formants, jitter, rng)
    _fade(out, n)
    write_wav(path, out, sample_rate)
    return {"kind": "vowel", "f0": f0, "formants": list(formants),
            "duration_s": duration, "sample_rate": sample_rate,
            "n_samples": n}


def _resonator_synth(n: int, sr: int, f0: float,
                     formants: tuple[float, ...], jitter: float,
                     rng: random.Random) -> list[float]:
    """声源-滤波器的时域实现：脉冲串过二阶带通谐振器。

    谐振器差分方程（Klatt 合成器的标准形式）：
        y[n] = A*x[n] + B*y[n-1] + C*y[n-2]
    其中 B、C 由中心频率和带宽决定。脉冲串 = 每过 1/f0 秒放一个
    单位脉冲，脉间加 ±0.5% 抖动。
    """
    # 每个谐振器：预计算系数
    resonators = []
    for fk in formants:
        bw = max(60.0, fk / 8.0)          # 带宽 ≈ F/8（经验值）
        r = math.exp(-math.pi * bw / sr)
        theta = 2 * math.pi * fk / sr
        # Klatt: C=-r², B=2r·cosθ, A=1-B-C（DC 增益归一）
        c = -r * r
        b = 2 * r * math.cos(theta)
        a = 1.0 - b - c
        resonators.append((a, b, c))

    out = []
    y1 = [0.0] * len(resonators)
    y2 = [0.0] * len(resonators)
    period = sr / f0
    next_pulse = 0.0
    for i in range(n):
        # 脉冲发生器：计数器过零发一个脉冲
        pulse = 0.0
        if i >= next_pulse:
            pulse = 1.0
            next_pulse = i + period * (1 + jitter * (rng.random() - 0.5) * 2)
        s = 0.0
        for ri, (a, b, c) in enumerate(resonators):
            y = a * pulse + b * y1[ri] + c * y2[ri]
            y2[ri] = y1[ri]
            y1[ri] = y
            s += y
        out.append(s)
    # 归一化到 0.8 满幅
    peak = max(abs(x) for x in out) or 1.0
    return [x * 0.8 / peak for x in out]


def synth_vowel_burst(path: str, sample_rate: int = 16000,
                      seed: int = 7) -> dict:
    """「元音-静音-元音」三段式：能量 VAD 的标准考题。

    中间 300ms 真零静音。VAD 断言：语音帧数 / 总帧数 ≈ 2/3±容差。
    """
    piece = 0.35
    silence = 0.30
    out: list[float] = []
    meta = {"pieces": 2, "silence_s": silence}
    for seg in range(2):
        part = _resonator_synth(
            int(sample_rate * piece), sample_rate, 130.0,
            (730, 1090, 2440), 0.005, random.Random(seed + seg))
        out.extend(part)
        if seg == 0:
            out.extend([0.0] * int(sample_rate * silence))
    _fade(out, len(out))
    write_wav(path, out, sample_rate)
    meta.update({"duration_s": 2 * piece + silence, "sample_rate": sample_rate,
                 "n_samples": len(out)})
    return meta


def synth_fricative_vs_vowel(path: str, sample_rate: int = 16000,
                             seed: int = 11) -> dict:
    """前半 /s/ 摩擦音（白噪声），后半元音：过零率特征的考题。

    摩擦音过零率高（随机符号翻转），浊音过零率低（周期信号）。
    断言：前半 ZCR > 后半 ZCR 的 2 倍。
    """
    rng = random.Random(seed)
    half = sample_rate // 2
    out = [rng.uniform(-0.5, 0.5) for _ in range(half)]
    out += _resonator_synth(half, sample_rate, 120.0, (730, 1090, 2440),
                            0.005, rng)
    _fade(out, len(out))
    write_wav(path, out, sample_rate)
    return {"kind": "fricative+ vowel", "half_samples": half,
            "duration_s": 1.0, "sample_rate": sample_rate,
            "n_samples": len(out)}


def synth_two_tones(path: str, sample_rate: int = 16000,
                    f_low: float = 440.0, f_high: float = 880.0,
                    duration: float = 0.8) -> dict:
    """双正弦交替：频谱质心的考题。

    前半 440Hz（质心 ≈ 440），后半 880Hz（质心 ≈ 880），
    断言两段质心比 ≈ 2（±10%）。
    """
    n = int(sample_rate * duration)
    half = n // 2
    out = [0.9 * math.sin(2 * math.pi * f_low * i / sample_rate)
           for i in range(half)]
    out += [0.9 * math.sin(2 * math.pi * f_high * i / sample_rate)
            for i in range(n - half)]
    _fade(out, n)
    write_wav(path, out, sample_rate)
    return {"kind": "tones", "f_low": f_low, "f_high": f_high,
            "duration_s": duration, "sample_rate": sample_rate,
            "n_samples": n}


def synth_chirp(path: str, sample_rate: int = 16000,
                f_start: float = 200.0, f_end: float = 3000.0,
                duration: float = 1.0) -> dict:
    """线性扫频：语谱图可视化的考题 —— 一条从左下到右上的斜线，
    看错窗长/窗移立刻现形。"""
    n = int(sample_rate * duration)
    out = []
    phase = 0.0
    for i in range(n):
        f = f_start + (f_end - f_start) * i / n
        phase += 2 * math.pi * f / sample_rate
        out.append(0.8 * math.sin(phase))
    _fade(out, n)
    write_wav(path, out, sample_rate)
    return {"kind": "chirp", "f_start": f_start, "f_end": f_end,
            "duration_s": duration, "sample_rate": sample_rate,
            "n_samples": n}


SIGNALS = {
    "vowel_a": synth_vowel,
    "vowel_burst": synth_vowel_burst,
    "fricative_vowel": synth_fricative_vs_vowel,
    "two_tones": synth_two_tones,
    "chirp": synth_chirp,
}

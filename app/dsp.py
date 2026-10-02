"""基础信号处理原语：全部纯标准库，不碰 numpy。

为什么不用 numpy：这个项目是「语音信号处理实验台」，定位是把
DSP 的每一步摊开给人看 —— 预加重到底乘了什么系数、窗函数长什么样、
FFT 之后取模的谱线怎么对上频率轴。numpy 一行 `np.fft.rfft` 会把
最值得学的那一层黑掉。而且核心零依赖意味着这个目录可以被整个拷进
任何 Python 环境直接用（包括嵌入式/教学机）。

代价是性能：radix-2 FFT 是纯 Python 循环，512 点约 0.5ms，
1 秒 16kHz 音频的全量语谱图（约 60 帧）在百毫秒级 —— 教学场景完全够用，
也写进了 e2e 的性能断言里。
"""

from __future__ import annotations

import cmath
import math


# ---------------------------------------------------------------- 序列工具

def next_pow2(n: int) -> int:
    """>= n 的最小 2 的幂。FFT 帧长必须是 2 的幂。"""
    p = 1
    while p < n:
        p <<= 1
    return p


def pre_emphasis(samples: list[float], coef: float = 0.97) -> list[float]:
    """一阶高通预加重：y[n] = x[n] - coef * x[n-1]。

    为什么要有这一步：语音的功率谱整体以约 -6dB/倍频程 下坠
    （辐射特性 + 声门谱倾斜），不补偿的话高频共振峰几乎埋在低频里。
    差分近似一阶导数，把谱抬平，让后面的共振峰/MFCC 公平对待每个频段。
    首样本没有前项，约定保持原值。
    """
    if not samples:
        return []
    out = [samples[0]]
    prev = samples[0]
    for x in samples[1:]:
        out.append(x - coef * prev)
        prev = x
    return out


def frame_signal(samples: list[float], frame_len: int, hop: int
                 ) -> list[list[float]]:
    """切成固定长度的帧，帧移 hop。最后一帧不足就丢弃（不补零：
    补零会让该帧能量虚假下降，画在能量曲线上是一个假低谷）。"""
    if frame_len <= 0 or hop <= 0:
        raise ValueError(f"frame_len/hop 必须为正，收到 {frame_len}/{hop}")
    if len(samples) < frame_len:
        raise ValueError(
            f"音频太短：{len(samples)} 样本放不进一帧（帧长 {frame_len}）")
    frames = []
    for start in range(0, len(samples) - frame_len + 1, hop):
        frames.append(samples[start:start + frame_len])
    return frames


def hamming(n: int) -> list[float]:
    """汉明窗：0.54 - 0.46*cos(2πk/(n-1))。

    为什么分帧要加窗：直接切帧等于矩形窗，频谱泄漏的旁瓣只有 -13dB，
    相邻频段的能量会互相污染。汉明窗旁瓣约 -41dB，代价是主瓣宽一倍 ——
    语谱图上时间/频率分辨率的折中就是从这里来的。
    """
    if n <= 1:
        return [1.0] * n
    return [0.54 - 0.46 * math.cos(2 * math.pi * k / (n - 1)) for k in range(n)]


# ---------------------------------------------------------------- FFT

def fft(x: list[float]) -> list[complex]:
    """radix-2 迭代式 Cooley-Tukey FFT。

    实现选择：
    - **迭代**而不是递归：递归版每层都要切一半复制列表，Python 对象
      开销大；迭代位反转版原地蝶形，内存和常数都小一截。
    - 长度必须是 2 的幂 —— 调用方负责把帧补到 next_pow2。
      教学实现只做一种基，代码可读性优先。
    """
    n = len(x)
    if n <= 0 or n & (n - 1):
        raise ValueError(f"FFT 长度必须是 2 的幂，收到 {n}")
    if n == 1:
        return [complex(x[0])]

    # 位反转置换：把下标按二进制倒排，蝶形就能自底向上原地做
    bits = n.bit_length() - 1
    a = [complex(v) for v in x]
    for i in range(n):
        j = int(f"{i:0{bits}b}"[::-1], 2)
        if j > i:
            a[i], a[j] = a[j], a[i]

    size = 2
    while size <= n:
        half = size // 2
        # 旋转因子 w = e^(-2πik/size)，用 cmath.exp 而不是查表：
        # 表要预生成且浮点误差会累积，直接算在这个规模下更快更准
        step = -2 * math.pi / size
        for start in range(0, n, size):
            for k in range(half):
                w = cmath.exp(step * k * 1j)
                u = a[start + k]
                v = a[start + k + half] * w
                a[start + k] = u + v
                a[start + k + half] = u - v
        size *= 2
    return a


def power_spectrum(frame: list[float], fft_len: int | None = None
                   ) -> list[float]:
    """单帧功率谱：|X[k]|²，只返回 0..Nyquist 的前一半。

    输入长度不足 fft_len 时末尾补零（补零不改变谱的形状，
    只是把插值网格变密 —— 这正是「窗长决定频率分辨率」的直观证据）。
    """
    n = fft_len or next_pow2(len(frame))
    buf = list(frame) + [0.0] * (n - len(frame))
    spec = fft(buf)
    half = n // 2 + 1
    return [abs(spec[k]) ** 2 for k in range(half)]


# ---------------------------------------------------------------- 帧级特征

def energy(frame: list[float]) -> float:
    """帧短时能量（不除以 N：绝对能量本来就和窗长成正比，保持物理直觉）。"""
    return sum(x * x for x in frame)


def zero_crossing_rate(frame: list[float], eps: float = 1e-12) -> int:
    """过零率：相邻样本符号翻转的次数。

    eps 是死区：数字录音的底噪会在零附近抖出大量假过零，
    |x| <= eps 的样本不参与判定。默认死区取 1e-12（纯合成信号够用），
    真实录音可以调大到量化步长的一半。
    """
    count = 0
    for a, b in zip(frame, frame[1:]):
        if abs(a) <= eps or abs(b) <= eps:
            continue
        if (a > 0) != (b > 0):
            count += 1
    return count


def autocorrelation_pitch(frame: list[float], sample_rate: int,
                          fmin: float = 60.0, fmax: float = 400.0
                          ) -> tuple[float, float]:
    """自相关法基频估计。返回 (频率 Hz, 置信度 0~1)。

    原理：周期信号的自相关在周期整数倍处出现峰。在候选周期范围内
    找最高峰，峰值越接近 1（归一化后）说明周期性越强。

    为什么不用 FFT 找谱峰：基频可能不在谱峰上（奇次谐波缺失、
    共振峰压低基波），自相关在时域直接对齐周期，对此免疫。

    细节：
    - 候选滞后区间 [sr/fmax, sr/fmin]，两端都验；
    - 归一化用滞后 0 处的能量（而不是帧能量），帧末尾没参与
      对齐的部分会压低所有候选峰，但压得均匀，不影响 argmax；
    - 没找到峰（静音/纯噪声）返回 (0.0, 0.0)。
    """
    n = len(frame)
    lo = max(1, int(sample_rate / fmax))
    hi = min(n - 1, int(sample_rate / fmin))
    if hi <= lo:
        return 0.0, 0.0

    e0 = sum(x * x for x in frame)
    if e0 <= 0:
        return 0.0, 0.0

    best_lag, best_val = 0, -1.0
    for lag in range(lo, hi + 1):
        r = sum(frame[i] * frame[i + lag] for i in range(n - lag))
        if r > best_val:
            best_val, best_lag = r, lag

    confidence = max(0.0, min(1.0, best_val / e0))
    if confidence < 0.30:
        # 峰太低：不是周期信号（噪声 / 静音），报无基频
        return 0.0, confidence
    freq = sample_rate / best_lag
    return freq, confidence


def spectral_centroid(spec: list[float], sample_rate: int, fft_len: int
                      ) -> float:
    """频谱质心（Hz）：功率加权的平均频率，「音色亮暗」的粗略度量。"""
    total = sum(spec)
    if total <= 0:
        return 0.0
    weighted = sum(f * p for f, p in zip(
        (k * sample_rate / fft_len for k in range(len(spec))), spec))
    return weighted / total

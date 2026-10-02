"""特征提取：MFCC、基频轮廓、VAD。叠在 dsp.py 的原语之上。

MFCC（Mel 频率倒谱系数）的流水线，每一步都对应一段人类听觉知识：

    分帧加窗 → FFT 功率谱 → Mel 滤波器组（低频密高频疏，模拟耳蜗）
    → 取对数（响度感觉与声强成对数，Weber-Fechner 定律）
    → DCT-II 去相关（取前 13 个系数，0 号是能量，1-12 是音色）

纯 Python 实现，重点是把每个中间量算得**正确且可解释**，
速度不是目标（见 dsp.py 的说明）。
"""

from __future__ import annotations

import math

from .dsp import (autocorrelation_pitch, energy, frame_signal, hamming,
                  power_spectrum, pre_emphasis, zero_crossing_rate)


def hz_to_mel(f: float) -> float:
    """Hz → Mel。用 Slaney/htk 常见的 2595*log10(1+f/700)。"""
    return 2595.0 * math.log10(1.0 + f / 700.0)


def mel_to_hz(m: float) -> float:
    """Mel → Hz。hz_to_mel 的逆。"""
    return 700.0 * (10 ** (m / 2595.0) - 1.0)


def mel_filterbank(n_filters: int, fft_len: int, sample_rate: int,
                   fmin: float = 50.0, fmax: float | None = None
                   ) -> list[list[float]]:
    """Mel 三角滤波器组。返回 [n_filters][fft_len//2+1] 的权重矩阵。

    实现细节：三角形的底边跨在 Mel 轴的等距点上，换算回 Hz 后
    低频滤波器窄、高频宽 —— 这就是「耳蜗低频分辨率高」的离散化。
    fmax 默认取 Nyquist；fmin 默认 50Hz（低于人声基频下限，
    滤了工频哼声）。
    """
    n_bins = fft_len // 2 + 1
    fmax = fmax if fmax is not None else sample_rate / 2.0
    if not (0 <= fmin < fmax <= sample_rate / 2):
        raise ValueError(
            f"频率范围不合法：{fmin}..{fmax} @ {sample_rate}Hz")

    mel_pts = [hz_to_mel(fmin + (fmax - fmin) * i / (n_filters + 2))
               for i in range(n_filters + 2)]
    hz_pts = [mel_to_hz(m) for m in mel_pts]
    bin_pts = [int(round(h * (fft_len + 1) / sample_rate)) for h in hz_pts]

    fb: list[list[float]] = []
    for i in range(n_filters):
        lo, cen, hi = bin_pts[i], bin_pts[i + 1], bin_pts[i + 2]
        weights = [0.0] * n_bins
        for k in range(lo, min(cen, n_bins)):
            if cen > lo:
                weights[k] = (k - lo) / (cen - lo)
        for k in range(cen, min(hi, n_bins)):
            if hi > cen:
                weights[k] = (hi - k) / (hi - cen)
        fb.append(weights)
    return fb


def dct2(x: list[float], n_out: int) -> list[float]:
    """DCT-II（正交归一化），只取前 n_out 个系数。

    直接按定义算 O(N²)：MFCC 只对 ~26 个 Mel 能量做 DCT，
    N 小到 O(N²) 毫无压力；且正交归一（k=0 乘 sqrt(1/4N)）
    保证系数能量可比。
    """
    n = len(x)
    out = []
    for k in range(n_out):
        s = 0.0
        for i, v in enumerate(x):
            s += v * math.cos(math.pi * k * (2 * i + 1) / (2 * n))
        scale = math.sqrt(1.0 / (4 * n)) if k == 0 else math.sqrt(2.0 / n)
        out.append(s * scale)
    return out


def mfcc(frames: list[list[float]], sample_rate: int,
         n_mfcc: int = 13, n_filters: int = 26, fft_len: int = 512,
         preemph: float = 0.97) -> list[list[float]]:
    """整段语音的 MFCC 序列。返回 [帧数][n_mfcc]。

    用了三次与教科书不同的工程决定，都写清楚理由：
    1. 每帧先去直流（减均值）：录音偏置会让 0 号 Mel 滤波器被抬高；
    2. Mel 能量取 log 时下限钳在 1e-10：真 0 能量会 log 爆炸；
    3. 0 号系数不替换成帧能量对数 —— 保留完整 DCT 输出，
       让「0 号≈响度、1-12≈音色」的教学对应关系可见。
    """
    win = hamming(len(frames[0])) if frames else []
    fb = mel_filterbank(n_filters, fft_len, sample_rate)
    out = []
    for fr in frames:
        # 预加重在帧内做（和加窗同一遍循环，省一次全序列拷贝）
        emphasized = [fr[0]] + [fr[i] - preemph * fr[i - 1]
                                for i in range(1, len(fr))]
        dc = sum(emphasized) / len(emphasized)
        windowed = [(e - dc) * w for e, w in zip(emphasized, win)]
        spec = power_spectrum(windowed, fft_len)
        mel_e = []
        for weights in fb:
            e = sum(w * p for w, p in zip(weights, spec))
            mel_e.append(math.log(max(e, 1e-10)))
        out.append(dct2(mel_e, n_mfcc))
    return out


def vad_mask(frames: list[list[float]], sample_rate: int,
             threshold_db: float = -45.0) -> list[bool]:
    """能量 VAD：帧能量高于阈值即语音。

    阈值是绝对 dBFS —— 简单，但要求录音电平大致正常。
    更稳的做法是自适应双阈值（Otsu/高斯混合），对教学演示过重，
    这里选择诚实声明局限。
    """
    mask = []
    for fr in frames:
        e = energy(fr)
        db = 10 * math.log10(max(e, 1e-20))
        mask.append(db >= threshold_db)
    return mask


def pitch_track(frames: list[list[float]], sample_rate: int,
                voiced: list[bool] | None = None
                ) -> list[tuple[float, float]]:
    """基频轮廓：每帧 (Hz, 置信度)，非浊音帧 (0, conf)。

    自相关法对帧长敏感：要覆盖 60Hz（16.7ms 周期）至少需要
    两个周期 ≈ 33ms。调用方应保证帧长 ≥ 2.5 * sr/fmin。
    """
    return [autocorrelation_pitch(fr, sample_rate) if (voiced is None or v)
            else (0.0, 0.0)
            for fr, v in zip(frames, voiced or [True] * len(frames))]


class FrameAnalysis:
    """一次完整分析的中间产物，供 SVG 渲染与 JSON 序列化共用。"""

    def __init__(self, samples: list[float], sample_rate: int,
                 frame_len_ms: float = 25.0, hop_ms: float = 10.0,
                 fft_len: int = 512):
        self.sample_rate = sample_rate
        self.n_samples = len(samples)
        self.frame_len = int(sample_rate * frame_len_ms / 1000)
        self.hop = int(sample_rate * hop_ms / 1000)
        if self.frame_len < 1 or self.hop < 1:
            raise ValueError("帧长/帧移至少 1 个样本：采样率太低或毫秒太小")
        if self.frame_len > len(samples):
            raise ValueError(
                f"音频太短：{self.n_samples} 样本放不进一帧 "
                f"（帧长 {self.frame_len} = {frame_len_ms}ms @ {sample_rate}Hz）")

        emphasized = pre_emphasis(samples)
        self.frames = frame_signal(emphasized, self.frame_len, self.hop)
        self.win = hamming(self.frame_len)
        self.fft_len = fft_len

        self.energies = [energy(fr) for fr in self.frames]
        self.zcrs = [zero_crossing_rate(fr) for fr in self.frames]
        self.voiced = vad_mask(self.frames, sample_rate)
        self.pitch = pitch_track(self.frames, sample_rate, self.voiced)
        self.mfcc = mfcc(self.frames, sample_rate, fft_len=fft_len)
        self.centroids = [
            _centroid_of(fr, self.win, fft_len, sample_rate)
            for fr in self.frames]

    # ---- 频谱图数据 -----------------------------------------------------
    def spectrogram(self) -> list[list[float]]:
        """[帧][频率bin] 的功率谱（dB），给 SVG 语谱图着色用。"""
        import math as m
        frames_db = []
        for fr in self.frames:
            windowed = [s * w for s, w in zip(fr, self.win)]
            spec = power_spectrum(windowed, self.fft_len)
            frames_db.append([10 * m.log10(max(p, 1e-12)) for p in spec])
        return frames_db


def _centroid_of(frame: list[float], win: list[float], fft_len: int,
                 sample_rate: int) -> float:
    from .dsp import spectral_centroid, power_spectrum
    windowed = [s * w for s, w in zip(frame, win)]
    spec = power_spectrum(windowed, fft_len)
    return spectral_centroid(spec, sample_rate, fft_len)

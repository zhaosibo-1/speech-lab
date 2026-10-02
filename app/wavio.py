"""WAV 读写：只支持 16-bit PCM（教学约定，够覆盖所有演示场景）。

不用 scipy.io.wavfile —— 同样是零依赖原则。标准库 wave 模块
读出的原始数据是 bytes，这里用 struct 一次解包成 list[float]
（归一化到 -1..1），上游所有 DSP 都只认这个格式。
"""

from __future__ import annotations

import math
import struct
import wave


class WavError(ValueError):
    """格式不支持。给 API 层一个可识别的异常类型。"""


def read_wav(path: str) -> tuple[list[float], int]:
    """读 16-bit PCM WAV，返回 (样本[-1..1], 采样率)。

    多声道交错存储 —— 这里做平均降为单声道（语音分析的标准做法，
    相位信息对特征没用）。
    """
    with wave.open(path, "rb") as w:
        nch = w.getnchannels()
        width = w.getsampwidth()
        rate = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    if width != 2:
        raise WavError(
            f"只支持 16-bit PCM（sampwidth=2），收到 sampwidth={width}。"
            "其他位深请先转换")
    data = struct.unpack(f"<{n * nch}h", raw)
    if nch > 1:
        mono = [sum(data[i:i + nch]) / nch for i in range(0, len(data), nch)]
    else:
        mono = list(data)
    samples = [x / 32768.0 for x in mono]
    return samples, rate


def write_wav(path: str, samples: list[float], sample_rate: int) -> None:
    """把 [-1..1] 的浮点样本写成 16-bit 单声道 WAV。越界样本裁剪。"""
    clipped = []
    for x in samples:
        if x > 1.0:
            x = 1.0
        elif x < -1.0:
            x = -1.0
        # 32767 而不是 32768：+32768 会溢出到 -32768
        clipped.append(int(round(x * 32767.0)))
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(struct.pack(f"<{len(clipped)}h", *clipped))


def resample_linear(samples: list[float], src_rate: int, dst_rate: int
                    ) -> list[float]:
    """线性插值重采样。

    只用于演示/归一化场景（例如把不同采样率的录音统一到 16k 再比特征）。
    线性插值的镜像抗混叠很差 —— 高倍降采样前应该先低通，
    这里不做，写清楚适用边界即可。
    """
    if src_rate == dst_rate or not samples:
        return list(samples)
    ratio = dst_rate / src_rate
    out_len = int(round(len(samples) * ratio))
    out = []
    for j in range(out_len):
        pos = j / ratio
        i = int(pos)
        frac = pos - i
        a = samples[i] if i < len(samples) else samples[-1]
        b = samples[i + 1] if i + 1 < len(samples) else a
        out.append(a + (b - a) * frac)
    return out


def seconds(n: int, sample_rate: int) -> float:
    """样本数 → 秒（保留 3 位，纯格式化辅助）。"""
    return round(n / sample_rate, 3)


def dbfs(samples: list[float]) -> float:
    """峰值电平 dBFS（相对满幅 1.0）。静音约定 -inf 太吵，报 -120。"""
    peak = max((abs(x) for x in samples), default=0.0)
    if peak <= 0:
        return -120.0
    return round(20 * math.log10(peak), 1)

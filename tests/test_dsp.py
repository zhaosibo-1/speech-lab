"""dsp.py 数值正确性：每个 DSP 原语都对着可手算的参考值验证。

这些测试是这个项目可信度的基石 —— FFT 写错一行，
后面所有特征都在「漂亮地错着」。
"""

import math

import pytest

from app.dsp import (autocorrelation_pitch, energy, fft, frame_signal,
                     hamming, next_pow2, power_spectrum, pre_emphasis,
                     spectral_centroid, zero_crossing_rate)


class TestNextPow2:
    def test_examples(self):
        assert next_pow2(1) == 1
        assert next_pow2(2) == 2
        assert next_pow2(3) == 4
        assert next_pow2(400) == 512
        assert next_pow2(513) == 1024


class TestPreEmphasis:
    def test_constant_signal_attenuated_by_dc_gain(self):
        """直流经一阶高通后衰减到 (1-coef) 倍，不是 0。

        完全去直流靠的是 MFCC 里的"减均值"步骤；
        预加重只负责把 -6dB/倍频程 的谱斜率抬平。
        把两者混为一谈是最常见的概念错误，这里钉死。"""
        out = pre_emphasis([0.5] * 10, 0.97)
        assert out[0] == 0.5
        assert all(v == pytest.approx(0.5 * 0.03, abs=1e-12) for v in out[1:])

    def test_formula(self):
        out = pre_emphasis([1.0, 0.8, 0.6], 0.97)
        assert out[1] == pytest.approx(0.8 - 0.97, abs=1e-12)
        assert out[2] == pytest.approx(0.6 - 0.97 * 0.8, abs=1e-12)

    def test_empty(self):
        assert pre_emphasis([]) == []


class TestFraming:
    def test_frames_and_remainder_dropped(self):
        # 10 样本，帧长 4，帧移 2 → 起点 0,2,4,6 → 4 帧；
        # [6,7,8,9] 恰好完整，没有样本被丢
        frames = frame_signal(list(range(10)), 4, 2)
        assert len(frames) == 4
        assert frames[0] == [0, 1, 2, 3]
        assert frames[-1] == [6, 7, 8, 9]

    def test_remainder_dropped(self):
        # 11 样本同参数 → 起点 0,2,4,6；末尾 1 个样本不足一帧，丢弃
        frames = frame_signal(list(range(11)), 4, 2)
        assert len(frames) == 4
        assert frames[-1] == [6, 7, 8, 9]

    def test_shorter_than_frame_raises(self):
        with pytest.raises(ValueError, match="帧长"):
            frame_signal([1.0, 2.0], 4, 2)

    def test_no_zero_padding(self):
        """不补零：最后一帧能量不会因为补零而虚假下降。"""
        frames = frame_signal([1.0] * 7, 4, 2)
        assert all(energy(f) == 4.0 for f in frames)


class TestHamming:
    def test_symmetric_and_positive(self):
        w = hamming(64)
        assert len(w) == 64
        assert w[0] == pytest.approx(0.08, abs=1e-9)   # 0.54-0.46
        assert w[-1] == pytest.approx(0.08, abs=1e-9)
        assert all(v > 0 for v in w)

    def test_peak_near_center(self):
        w = hamming(64)
        # cos(2πk/63) 在 k=32 处还没回到 1，峰是 0.9994 不是 1.0 ——
        # 汉明窗精确到峰需要奇数窗长；差 6e-4 是公式本身的性质
        assert max(w) == pytest.approx(1.0, abs=1e-3)
        assert w.index(max(w)) in (31, 32)


class TestFFT:
    def test_dc_only(self):
        """全 1 信号 → 只有 0 号 bin 有值，且等于 N。"""
        spec = fft([1.0] * 8)
        assert abs(spec[0]) == pytest.approx(8.0)
        assert all(abs(s) < 1e-9 for s in spec[1:])

    def test_single_tone_lands_on_bin(self):
        """f = 3·fs/N 的正弦 → 能量集中在 3 号 bin（和它的镜像 5 号）。"""
        n = 16
        x = [math.sin(2 * math.pi * 3 * i / n) for i in range(n)]
        spec = fft(x)
        assert abs(spec[3]) == pytest.approx(n / 2, rel=1e-9)
        assert abs(spec[13]) == pytest.approx(n / 2, rel=1e-9)
        assert abs(spec[0]) < 1e-9

    def test_parseval(self):
        """Parseval 定理：时域能量 = 频域能量 / N。任何 FFT 实现的硬标准。"""
        n = 64
        x = [math.sin(2 * math.pi * 5 * i / n) + 0.3 for i in range(n)]
        time_energy = sum(v * v for v in x)
        spec = fft(x)
        freq_energy = sum(abs(s) ** 2 for s in spec) / n
        assert freq_energy == pytest.approx(time_energy, rel=1e-9)

    def test_matches_naive_dft(self):
        """和 O(N²) 的朴素 DFT 定义对拍 —— 两条完全不同的路径算出同一个答案。"""
        n = 32
        x = [math.cos(2 * math.pi * 7 * i / n) * (i % 5) for i in range(n)]
        spec = fft(x)
        for k in range(n):
            ref = sum(x[i] * cmath_exp(-2j * math.pi * k * i / n)
                      for i in range(n))
            assert abs(spec[k]) == pytest.approx(abs(ref), rel=1e-9)

    def test_rejects_non_pow2(self):
        with pytest.raises(ValueError, match="2 的幂"):
            fft([0.0] * 12)


def cmath_exp(z):
    import cmath
    return cmath.exp(z)


class TestPowerSpectrum:
    def test_half_spectrum_and_zero_pad(self):
        # 100 样本补到 128 点 FFT → 65 个 bin（0..Nyquist）
        x = [0.1 * (i % 3) for i in range(100)]
        spec = power_spectrum(x)
        assert len(spec) == 65
        assert all(p >= 0 for p in spec)

    def test_tone_frequency_mapping(self):
        sr, n = 1000, 256
        f = 100.0                                  # 对应 bin 25.6
        x = [math.sin(2 * math.pi * f * i / sr) for i in range(n)]
        spec = power_spectrum(x, n)
        peak_bin = max(range(len(spec)), key=lambda k: spec[k])
        assert peak_bin == 26                      # 100/1000*256 = 25.6 → 26


class TestFrameFeatures:
    def test_energy_of_tone(self):
        n = 100
        x = [math.sin(2 * math.pi * i / n) for i in range(n)]
        assert energy(x) == pytest.approx(n / 2, rel=1e-9)  # sin² 均值 1/2

    def test_zcr_noise_beats_tone(self):
        import random
        rng = random.Random(3)
        noise = [rng.choice([-1.0, 1.0]) for _ in range(200)]
        tone = [math.sin(2 * math.pi * i / 50) for i in range(200)]
        assert zero_crossing_rate(noise) > zero_crossing_rate(tone) * 2

    def test_zcr_dead_zone(self):
        """死区内的抖动不算过零：底噪不该制造假过零。"""
        jitter = [1e-13, -1e-13] * 50
        assert zero_crossing_rate(jitter) == 0
        assert zero_crossing_rate(jitter, eps=0.0) == 99


class TestPitch:
    def test_synthetic_f0_recovered(self, vowel):
        """最硬的一条：合成 /a/ 用 120Hz，追踪必须报 120±5。"""
        samples, sr = vowel
        frames = frame_signal(samples, int(sr * 0.04), int(sr * 0.01))
        freqs = [autocorrelation_pitch(fr, sr)[0] for fr in frames]
        voiced = [f for f in freqs if f > 0]
        assert voiced, "120Hz 周期信号必须有浊音帧"
        assert all(abs(f - 120) < 5 for f in voiced)

    def test_silence_gives_no_pitch(self):
        f, conf = autocorrelation_pitch([0.0] * 400, 16000)
        assert f == 0.0 and conf == 0.0

    def test_aperiodic_rejected(self):
        import random
        rng = random.Random(9)
        noise = [rng.uniform(-1, 1) for _ in range(640)]
        f, conf = autocorrelation_pitch(noise, 16000)
        assert f == 0.0          # 白噪声不该被报成有基频
        assert conf < 0.3

    def test_out_of_range_high_pitch_reports_subharmonic(self):
        """800Hz 超出 fmax=400 → 报出它的次谐波 400Hz，而不是 0。

        这是自相关法的**已知局限**：lag = sr/f0 * 2 处自相关同样出现
        高峰（两个完整周期对齐）。搜索范围里只有次谐波峰时它就会被选中。
        生产实现会再用 lag 倍频程比较或 cepstrum 消歧；
        教学实现选择把这个行为钉死在测试里，让局限可见。
        """
        sr = 16000
        x = [math.sin(2 * math.pi * 800 * i / sr) for i in range(640)]
        f, _ = autocorrelation_pitch(x, sr, fmin=60, fmax=400)
        assert f == pytest.approx(400.0, abs=1.0)


class TestCentroid:
    def test_low_tone_lower_centroid_than_high(self):
        sr, n = 8000, 512
        def centroid_of(f):
            x = [math.sin(2 * math.pi * f * i / sr) for i in range(n)]
            return spectral_centroid(power_spectrum(x, n), sr, n)
        assert centroid_of(500) < centroid_of(2000)

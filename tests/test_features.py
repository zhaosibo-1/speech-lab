"""features.py：Mel 滤波器组、DCT、MFCC、VAD 的数值验证。"""

import math

import pytest

from app.dsp import frame_signal
from app.features import (FrameAnalysis, dct2, hz_to_mel, mel_filterbank,
                          mel_to_hz, mfcc, vad_mask)
from app.wavio import read_wav


class TestMel:
    def test_roundtrip(self):
        for f in (0.0, 100.0, 1000.0, 4000.0, 7900.0):
            assert mel_to_hz(hz_to_mel(f)) == pytest.approx(f, rel=1e-9)

    def test_reference_values(self):
        """2595*log10(1+f/700) 的两个手算锚点。"""
        assert hz_to_mel(700) == pytest.approx(2595 * math.log10(2))
        assert hz_to_mel(0) == 0.0

    def test_filterbank_shape(self):
        fb = mel_filterbank(26, 512, 16000)
        assert len(fb) == 26
        assert all(len(row) == 257 for row in fb)   # 512//2+1

    def test_triangles_are_consecutive(self):
        """每个滤波器的支撑区间应与相邻滤波器首尾相接（Mel 轴等距）。"""
        fb = mel_filterbank(10, 512, 16000)
        for a, b in zip(fb, fb[1:]):
            a_max = max(i for i, v in enumerate(a) if v > 0)
            b_min = min(i for i, v in enumerate(b) if v > 0)
            assert b_min <= a_max + 1, "滤波器之间出现空洞"

    def test_each_filter_sums_positive(self):
        fb = mel_filterbank(26, 512, 16000)
        assert all(sum(row) > 0 for row in fb)

    def test_invalid_range(self):
        with pytest.raises(ValueError, match="频率范围"):
            mel_filterbank(26, 512, 16000, fmin=9000, fmax=4000)


class TestDCT2:
    def test_dc_input(self):
        """常数输入 → 只有 0 号系数非零（DCT 的能量集中性质）。"""
        out = dct2([2.0] * 26, 13)
        assert out[0] != 0
        assert all(abs(v) < 1e-9 for v in out[1:])

    def test_energy_compression(self):
        """平滑谱的前几个系数携带绝大部分能量 —— MFCC 为什么取前 13 维。"""
        x = [math.log(1 + 10 * math.exp(-i / 8)) for i in range(26)]
        out = dct2(x, 26)
        head = sum(v * v for v in out[:13])
        total = sum(v * v for v in out)
        assert head / total > 0.9


class TestMFCC:
    def test_shape_and_finiteness(self, vowel):
        samples, sr = vowel
        frames = frame_signal(samples, 400, 160)
        coeff = mfcc(frames, sr, n_mfcc=13)
        assert len(coeff) == len(frames)
        assert all(len(row) == 13 for row in coeff)
        assert all(math.isfinite(v) for row in coeff for v in row)

    def test_silent_frame_does_not_explode(self, vowel):
        """全零帧 log(0) 被 1e-10 钳住，输出仍是有限值。"""
        import random
        samples, sr = vowel
        frames = frame_signal(samples, 400, 160)
        frames[0] = [0.0] * 400
        coeff = mfcc(frames, sr)
        assert all(math.isfinite(v) for v in coeff[0])


class TestVAD:
    def test_burst_signal_voiced_ratio(self, tmp_wav_dir):
        """合成「元音-静音-元音」：2/3 的时长是语音，VAD 必须近似给出 2/3。"""
        from app.synth import synth_vowel_burst
        path = tmp_wav_dir / "burst.wav"
        synth_vowel_burst(str(path))
        samples, sr = read_wav(str(path))
        fa = FrameAnalysis(samples, sr)
        # 首尾淡入淡出 + 帧跨界，容差 0.08
        ratio = sum(fa.voiced) / len(fa.voiced)
        assert abs(ratio - 2 / 3) < 0.08

    def test_threshold_semantics(self):
        loud = [0.5] * 400                       # 能量 100 → 20dB
        quiet = [0.0] * 400
        mask = vad_mask([loud, quiet], 16000, threshold_db=-45)
        assert mask == [True, False]


class TestFrameAnalysis:
    def test_frame_params_from_ms(self, vowel):
        samples, sr = vowel
        fa = FrameAnalysis(samples, sr, frame_len_ms=25, hop_ms=10)
        assert fa.frame_len == int(sr * 0.025)
        assert fa.hop == int(sr * 0.01)

    def test_rejects_audio_shorter_than_frame(self):
        with pytest.raises(ValueError, match="音频太短"):
            FrameAnalysis([0.1] * 100, 16000, frame_len_ms=25)

    def test_spectrogram_shape(self, vowel):
        samples, sr = vowel
        fa = FrameAnalysis(samples, sr)
        spec = fa.spectrogram()
        assert len(spec) == len(fa.frames)
        assert all(len(row) == fa.fft_len // 2 + 1 for row in spec)

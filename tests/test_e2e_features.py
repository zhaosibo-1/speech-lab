"""端到端特征断言：合成什么就该测出什么 —— 检测算法正确性的最硬证据。"""

import statistics

import pytest

from app.features import FrameAnalysis
from app.synth import (synth_vowel, synth_vowel_burst, synth_two_tones)


@pytest.fixture(scope="module")
def burst(tmp_path_factory):
    path = tmp_path_factory.mktemp("e2e") / "burst.wav"
    synth_vowel_burst(str(path))
    samples, sr = __import__("app.wavio",
                             fromlist=["read_wav"]).read_wav(str(path))
    return samples, sr


class TestVowel:
    def test_pitch_matches_synth_f0(self, vowel):
        samples, sr = vowel
        fa = FrameAnalysis(samples, sr)
        voiced = [f for f, _ in fa.pitch if f > 0]
        assert len(voiced) >= len(fa.pitch) * 0.9
        assert statistics.mean(voiced) == pytest.approx(120, abs=5)

    def test_formant_shape_in_centroid(self, vowel):
        """/a/ 的共振峰在 1-3kHz，质心应落在几百到几千 Hz 之间。"""
        samples, sr = vowel
        fa = FrameAnalysis(samples, sr)
        c = statistics.mean(fa.centroids)
        assert 500 < c < 5000

    def test_mfcc_c0_is_loudness_proxy(self, burst):
        """0 号 MFCC 与帧能量正相关（都是响度的对数尺度代理）。

        用带静音段的 burst 信号：均匀信号上 c0 恒定，
        断言会退化成同义反复 —— 必须有响度差异才测得出相关。"""
        samples, sr = burst
        fa = FrameAnalysis(samples, sr)
        c0 = [row[0] for row in fa.mfcc]
        e = fa.energies
        mean_e = statistics.mean(e)
        loud = [i for i in range(len(e)) if e[i] > mean_e]
        quiet = [i for i in range(len(e)) if e[i] <= mean_e]
        assert statistics.mean(c0[i] for i in loud) > \
            statistics.mean(c0[i] for i in quiet)


class TestVowelBurst:
    def test_vad_finds_silence(self, burst):
        samples, sr = burst
        fa = FrameAnalysis(samples, sr)
        # 合成时 2/3 是语音（两端元音）、1/3 真零静音
        ratio = sum(fa.voiced) / len(fa.voiced)
        assert 0.55 <= ratio <= 0.85

    def test_pitch_absent_in_silence(self, burst):
        samples, sr = burst
        fa = FrameAnalysis(samples, sr)
        silent = [i for i, v in enumerate(fa.voiced) if not v]
        assert silent, "静音段必须被 VAD 找出来"
        assert all(f == 0.0 for f, _ in (fa.pitch[i] for i in silent))


class TestTwoTones:
    def test_centroid_ratio_matches_frequency_ratio(self, tones):
        """前半 440Hz 后半 880Hz：两段质心之比 ≈ 2。

        这是「频谱质心 = 信号频率」最直接的数值验证，
        比任何近似断言都硬。
        """
        samples, sr = tones
        fa = FrameAnalysis(samples, sr)
        half = len(fa.centroids) // 2
        low = statistics.mean(fa.centroids[:half])
        high = statistics.mean(fa.centroids[half:])
        assert high / low == pytest.approx(2.0, rel=0.15)

    def test_hop_changes_frame_count(self, tones):
        """帧移减半 → 帧数约翻倍（不严格 2 倍：丢尾规则）。"""
        samples, sr = tones
        a = FrameAnalysis(samples, sr, hop_ms=10)
        b = FrameAnalysis(samples, sr, hop_ms=5)
        assert abs(len(b.frames) - 2 * len(a.frames)) <= 3

"""wavio.py：WAV 读写循环、格式校验、重采样。"""

import struct
import wave

import pytest

from app.synth import synth_vowel
from app.wavio import WavError, dbfs, read_wav, resample_linear, write_wav


class TestRoundtrip:
    def test_write_then_read(self, tmp_path):
        samples = [0.0, 0.5, -0.5, 1.0, -1.0] * 100
        path = str(tmp_path / "a.wav")
        write_wav(path, samples, 16000)
        back, sr = read_wav(path)
        assert sr == 16000
        assert len(back) == len(samples)
        # 16-bit 量化误差 < 1/32767
        assert all(a == pytest.approx(b, abs=1 / 32767)
                   for a, b in zip(back, samples))

    def test_clipping(self, tmp_path):
        write_wav(str(tmp_path / "c.wav"), [2.0, -2.0], 8000)
        back, _ = read_wav(str(tmp_path / "c.wav"))
        assert back[0] == pytest.approx(1.0, abs=1e-4)
        assert back[1] == pytest.approx(-1.0, abs=1e-4)

    def test_no_overflow_at_full_scale(self, tmp_path):
        """+1.0 必须映射到 32767 而不是 32768（会溢出成 -32768）。"""
        write_wav(str(tmp_path / "f.wav"), [1.0], 8000)
        with wave.open(str(tmp_path / "f.wav"), "rb") as w:
            raw = w.readframes(1)
        assert struct.unpack("<h", raw)[0] == 32767


class TestReadValidation:
    def test_8bit_rejected(self, tmp_path):
        path = str(tmp_path / "8bit.wav")
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(1)
            w.setframerate(8000)
            w.writeframes(b"\x80" * 100)
        with pytest.raises(WavError, match="16-bit"):
            read_wav(path)

    def test_stereo_downmixed(self, tmp_path):
        path = str(tmp_path / "st.wav")
        with wave.open(path, "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(struct.pack("<4h", 32767, -32767, 32767, -32767))
        samples, _ = read_wav(path)
        assert len(samples) == 2                       # 两声道 → 2 帧
        assert samples[0] == pytest.approx(0.0, abs=1e-3)  # (1 + -1)/2


class TestResample:
    def test_same_rate_is_identity(self):
        x = [0.1, 0.2, 0.3]
        assert resample_linear(x, 16000, 16000) == x

    def test_half_rate_length(self):
        x = list(range(100))
        out = resample_linear([float(v) for v in x], 16000, 8000)
        assert len(out) == 50

    def test_double_rate_preserves_endpoints(self):
        x = [0.0, 1.0]
        out = resample_linear(x, 8000, 16000)
        assert out[0] == pytest.approx(0.0)
        assert out[-1] == pytest.approx(1.0)


class TestDbfs:
    def test_full_scale_and_silence(self):
        assert dbfs([1.0, -1.0]) == pytest.approx(0.0, abs=0.1)
        assert dbfs([0.0] * 10) == -120.0
        assert dbfs([]) == -120.0
        # 0.1 满幅 → -20dB
        assert dbfs([0.1]) == pytest.approx(-20.0, abs=0.1)


class TestSynthWavFiles:
    def test_synth_output_is_readable(self, tmp_path):
        path = str(tmp_path / "v.wav")
        meta = synth_vowel(path, duration=0.3)
        samples, sr = read_wav(path)
        assert sr == meta["sample_rate"]
        assert len(samples) == meta["n_samples"]
        assert max(abs(s) for s in samples) <= 1.0

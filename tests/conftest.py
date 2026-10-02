"""共享夹具：合成信号缓存在 session 级，避免每个用例重复合成。"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.synth import synth_vowel, synth_two_tones  # noqa: E402
from app.wavio import read_wav  # noqa: E402


@pytest.fixture(scope="session")
def tmp_wav_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("wav")


@pytest.fixture(scope="session")
def vowel(tmp_wav_dir):
    """合成 /a/：基频 120Hz、共振峰 (730,1090,2440)。"""
    path = tmp_wav_dir / "vowel.wav"
    synth_vowel(str(path), duration=0.8)
    return read_wav(str(path))


@pytest.fixture(scope="session")
def tones(tmp_wav_dir):
    """前半 440Hz、后半 880Hz 的双正弦交替。"""
    path = tmp_wav_dir / "tones.wav"
    synth_two_tones(str(path))
    return read_wav(str(path))

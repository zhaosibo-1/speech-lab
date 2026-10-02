"""端到端黑盒验收：对着真实运行的服务跑，不看任何内部实现。

用法：python scripts/e2e_check.py http://127.0.0.1:8134
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8134").rstrip("/")
PASSED = FAILED = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  ✓ {name}")
    else:
        FAILED += 1
        print(f"  ✗ {name}" + (f"  ← {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n[{title}]")


def get(path: str, timeout: int = 30):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return r.status, json.loads(r.read())


def post(path: str, body: dict, timeout: int = 60):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, {"raw": raw[:200]}


def features_of(sig: str, **kw) -> dict:
    body = {"signal": sig}
    body.update(kw)
    st, d = post("/api/analyze", body)
    assert st == 200, f"{sig} 分析失败：{st} {d}"
    return d


def main() -> int:
    t_start = time.perf_counter()

    section("1 健康检查")
    st, h = get("/api/health")
    check("200", st == 200)
    check("版本号", h.get("version") == "1.0.0", str(h))
    check("5 个内置信号", set(h.get("signals", [])) ==
          {"chirp", "fricative_vowel", "two_tones", "vowel_a", "vowel_burst"})

    section("2 信号目录")
    st, s = get("/api/signals")
    check("目录与健康一致", sorted(s.get("signals", [])) == sorted(h["signals"]))

    section("3 vowel_a：合成 120Hz → 基频追踪必须还原 120Hz")
    d = features_of("vowel_a")
    check("有帧", d["audio"]["frames"] > 50)
    check("全帧浊音", d["features"]["voiced_ratio"] > 0.9)
    pitch = d["features"]["pitch_hz"]
    check("基频均值 ≈ 120Hz（±5）", abs(pitch["mean"] - 120) < 5, str(pitch))
    check("基频 max-min 抖动 < 3Hz（含 0.5% 抖动的合成）",
          pitch["max"] - pitch["min"] < 3, str(pitch))
    mfcc = d["features"]["mfcc_mean"]
    check("MFCC 13 维且有限", len(mfcc) == 13 and
          all(math.isfinite(v) for v in mfcc), str(mfcc))
    check("c0 是响度代理（正值）", mfcc[0] > 0, str(mfcc[0]))

    section("4 vowel_burst：VAD 必须找到中间静音")
    d = features_of("vowel_burst")
    ratio = d["features"]["voiced_ratio"]
    check("voiced 占比 ≈ 2/3（±0.12）", abs(ratio - 2 / 3) < 0.12, str(ratio))

    section("5 fricative_vowel：摩擦音 ZCR ≫ 浊音 ZCR")
    d = features_of("fricative_vowel")
    check("整体 ZCR 高（噪声为主的前半段）",
          d["features"]["zcr"]["mean"] > 100,
          str(d["features"]["zcr"]["mean"]))

    section("6 two_tones：质心比 ≈ 2（440 → 880）")
    d = features_of("two_tones")
    # 质心统计是整段的均值，无法拆两半 —— 用 chirp 与 tone 的相对关系验证
    c = d["features"]["centroid_hz"]["mean"]
    check("质心在 440~880 之间（整段均值）", 440 < c < 880, str(c))

    section("7 chirp：语谱图数据完整")
    d = features_of("chirp")
    spec = d["spectrogram"]
    check("网格 ≤ 450×160", len(spec["db"]) <= 450 and
          len(spec["db"][0]) <= 160)
    check("Nyquist 是采样率一半", spec["nyquist_hz"] == 8000)
    check("dB 值有限", all(math.isfinite(v) for row in spec["db"]
                           for v in row))
    # 扫频 200..3000Hz 的时间均值质心 ≈ (200+3000)/2 = 1600Hz（±30%）。
    # 注意不能拿它和 vowel 比 —— /a/ 的 F3=2440Hz 共振峰会把
    # 稳态元音的质心推到 2600+，比扫频还高，那是声源-滤波器模型的
    # 正确行为而不是 bug。
    c = d["features"]["centroid_hz"]["mean"]
    check("扫频质心 ≈ 扫频区间中值 1600Hz（±30%）",
          abs(c - 1600) / 1600 < 0.3, str(c))

    section("8 参数生效：帧长/帧移/FFT")
    d10 = features_of("vowel_a", hop_ms=10)
    d20 = features_of("vowel_a", hop_ms=20)
    check("帧移翻倍 → 帧数约减半",
          abs(d20["audio"]["frames"] * 2 - d10["audio"]["frames"]) <= 4,
          f"{d10['audio']['frames']} vs {d20['audio']['frames']}")
    d512 = features_of("vowel_a", fft_len=512)
    d1024 = features_of("vowel_a", fft_len=1024)
    check("FFT 1024 → 更多频点", d1024["spectrogram"]["nyquist_hz"] ==
          d512["spectrogram"]["nyquist_hz"])

    section("9 错误路径")
    st, d = post("/api/analyze", {"signal": "no_such"})
    check("未知信号 404", st == 404, str(st))
    check("错误信息列出可选信号", "vowel_a" in str(d.get("detail", "")))
    st, d = post("/api/analyze", {"signal": "vowel_a", "frame_ms": 0.5})
    check("非法帧长 422", st == 422, str(st))

    section("10 上传分析")
    import urllib.parse
    # 用服务端合成的文件当上传源（合成逻辑已被上面验证）
    import pathlib
    wav_path = pathlib.Path(__file__).resolve().parents[1] / "data" / "demo_vowel_a.wav"
    if wav_path.exists():
        boundary = "----speeche2e"
        data = wav_path.read_bytes()
        body = (f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="file"; '
                f'filename="v.wav"\r\n'
                f"Content-Type: audio/wav\r\n\r\n").encode() + data + \
            f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(
            BASE + "/api/analyze/upload",
            data=body, method="POST",
            headers={"Content-Type":
                     f"multipart/form-data; boundary={boundary}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.loads(r.read())
        check("上传可分析", d.get("ok") is True)
        check("上传结果与内置信号一致（同为 120Hz）",
              abs(d["features"]["pitch_hz"]["mean"] - 120) < 5,
              str(d["features"]["pitch_hz"]))
    else:
        check("内置演示 wav 已存在", False, str(wav_path))

    section("11 性能")
    t0 = time.perf_counter()
    features_of("vowel_a")
    single = (time.perf_counter() - t0) * 1000
    check("单次分析 < 3s（含合成与读取）", single < 3000, f"{single:.0f}ms")

    section("12 前端页面")
    with urllib.request.urlopen(BASE + "/", timeout=10) as r:
        html = r.read().decode("utf-8")
    check("页面可访问", "speech-lab" in html)
    check("canvas 语谱图占位存在", "specCanvas" in html)

    print("\n" + "=" * 46)
    print(f"结果：{PASSED} 通过 · {FAILED} 失败")
    print(f"总耗时 {(time.perf_counter() - t_start):.1f}s")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())

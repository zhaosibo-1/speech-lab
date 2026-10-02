"""FastAPI 服务：内置演示信号 + WAV 上传分析。

分析结果分两路：JSON（特征统计，给程序用）
和 SVG（波形/语谱图/轮廓，给页面直接内联）。
"""

from __future__ import annotations

import math
import statistics
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import visual
from .features import FrameAnalysis
from .synth import SIGNALS
from .wavio import WavError, dbfs, read_wav

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR.parent / "data"
WEB_DIR = APP_DIR.parent / "web"

app = FastAPI(title="speech-lab", version="1.0.0")

# 允许的音频上限：10MB。合成信号都远小于此，主要挡恶意上传
MAX_UPLOAD = 10 * 1024 * 1024


class AnalyzeBody(BaseModel):
    signal: str = Field(description="内置信号名，见 /api/signals")
    frame_ms: float = Field(25.0, gt=1, le=100)
    hop_ms: float = Field(10.0, gt=0.5, le=50)
    fft_len: int = Field(512)


def _stats(values: list[float]) -> dict[str, float]:
    """一组逐帧数值的摘要统计。空序列返回 0，不抛异常。"""
    if not values:
        return {"mean": 0.0, "max": 0.0, "min": 0.0}
    return {"mean": round(statistics.mean(values), 2),
            "max": round(max(values), 2),
            "min": round(min(values), 2)}


def _analyze(samples: list[float], sample_rate: int, frame_ms: float,
             hop_ms: float, fft_len: int) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        fa = FrameAnalysis(samples, sample_rate, frame_ms, hop_ms, fft_len)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    elapsed = (time.perf_counter() - t0) * 1000

    n_frames = len(fa.frames)
    voiced_ratio = round(sum(fa.voiced) / max(1, n_frames), 3)
    pitch_values = [f for f, _ in fa.pitch if f > 0]

    result: dict[str, Any] = {
        "ok": True,
        "audio": {
            "sample_rate": sample_rate,
            "n_samples": fa.n_samples,
            "duration_s": round(fa.n_samples / sample_rate, 3),
            "peak_dbfs": dbfs(samples),
            "frames": n_frames,
            "frame_len": fa.frame_len,
            "hop": fa.hop,
            "fft_len": fa.fft_len,
        },
        "features": {
            "pitch_hz": _stats(pitch_values),
            "energy": _stats(fa.energies),
            "zcr": _stats([float(z) for z in fa.zcrs]),
            "centroid_hz": _stats(fa.centroids),
            "voiced_ratio": voiced_ratio,
            # MFCC 只给均值向量：整段音色的「名片」，教学上够用
            "mfcc_mean": [round(v, 2) for v in
                          (statistics.mean(col) for col in
                           zip(*fa.mfcc))] if fa.mfcc else [],
        },
        "elapsed_ms": round(elapsed, 1),
    }

    spec = fa.spectrogram()
    # 语谱图不生成 SVG —— 98 帧 × 250 bin 的热图 = 2 万多个 rect，
    # SVG 要 1.5MB，浏览器直接卡。改为下采样后发 dB 网格，
    # 前端 canvas 上色（渲染代码在 index.html，一样逐格可读）。
    spec_grid, grid_hop_frames = _downsample_spec(spec, 450, 160)
    # 色标相对化：峰值是 0dB 往下 60dB 动态范围。
    # 用绝对 floor（如 -90dBFS）在合成/录音间不可比 ——
    # FFT 功率谱没有统一标定，绝对 dB 会导致大面积饱和或全黑。
    peak_db = max(v for row in spec_grid for v in row)
    floor_db = peak_db - 60.0
    result["spectrogram"] = {
        "db": spec_grid,
        "nyquist_hz": sample_rate / 2,
        "frame_ms": frame_ms,
        "hop_ms": hop_ms,
        "grid_hop_frames": grid_hop_frames,
        "floor_db": floor_db,
        "peak_db": peak_db,
    }
    result["svg"] = {
        "waveform": visual.waveform(samples),
        "pitch": visual.contour(
            [f for f, _ in fa.pitch], color="#f59e0b", zero_line=True,
            unit_suffix=" Hz"),
        "energy": visual.contour(fa.energies, color="#34d399",
                                 unit_suffix=""),
        "zcr": visual.contour([float(z) for z in fa.zcrs],
                              color="#60a5fa", unit_suffix=""),
        "centroid": visual.contour(fa.centroids, color="#c084fc",
                                   unit_suffix=" Hz"),
    }
    return result


def _downsample_spec(spec_db: list[list[float]], max_cols: int,
                     max_rows: int) -> tuple[list[list[float]], int]:
    """语谱图网格下采样：帧方向按列聚合取 max（峰值优先，避免
    短促瞬态在缩格子时被平均掉），频率方向截断到前 max_rows 个 bin。

    返回 (网格, 每格对应多少帧)。频率 bin 从低到高排列 ——
    Nyquist 8kHz 时 160 bin ≈ 50Hz/格，共振峰尺度足够。
    """
    n_frames = len(spec_db)
    cols = max(1, min(max_cols, n_frames))
    per = max(1, n_frames // cols)
    grid = []
    for c in range(cols):
        chunk = spec_db[c * per:(c + 1) * per]
        if not chunk:
            break
        grid.append([max(frame[b] for frame in chunk)
                     for b in range(min(max_rows, len(chunk[0])))])
    return grid, per


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "version": "1.0.0",
            "signals": sorted(SIGNALS)}


@app.get("/api/signals")
def signals() -> dict[str, Any]:
    """内置信号目录：名字 + 参数（不立即合成，合成在分析时做）。"""
    return {"signals": sorted(SIGNALS)}


@app.post("/api/analyze")
def analyze(body: AnalyzeBody) -> dict[str, Any]:
    if body.signal not in SIGNALS:
        raise HTTPException(
            status_code=404,
            detail=f"未知信号 {body.signal!r}，可选：{sorted(SIGNALS)}")
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"demo_{body.signal}.wav"
    SIGNALS[body.signal](str(path))
    samples, rate = read_wav(str(path))
    return _analyze(samples, rate, body.frame_ms, body.hop_ms, body.fft_len)


@app.post("/api/analyze/upload")
async def analyze_upload(file: UploadFile = File(...),
                         frame_ms: float = 25.0, hop_ms: float = 10.0,
                         fft_len: int = 512) -> dict[str, Any]:
    raw = await file.read()
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(status_code=413,
                            detail="文件超过 10MB 上限")
    tmp = DATA_DIR / "upload_tmp.wav"
    DATA_DIR.mkdir(exist_ok=True)
    tmp.write_bytes(raw)
    try:
        samples, rate = read_wav(str(tmp))
    except (WavError, Exception) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"WAV 解析失败：{exc}（只支持 16-bit PCM）") from exc
    return _analyze(samples, rate, frame_ms, hop_ms, fft_len)


@app.get("/")
def index_page() -> FileResponse:
    return FileResponse(f"{WEB_DIR}/index.html")


app.mount("/", StaticFiles(directory=WEB_DIR), name="web")

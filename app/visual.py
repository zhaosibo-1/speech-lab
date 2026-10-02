"""SVG 渲染：波形、语谱图、基频/能量轮廓。

为什么是 SVG 而不是 PNG：不用 Pillow/matplotlib（零依赖原则），
而 SVG 是文本，可以直接内联进 HTML、可以无损缩放，
浏览器端不需要任何 JS 图表库。渲染逻辑刻意朴素 ——
就是「数据坐标 → 视口坐标」的线性映射加矩形/折线，
让学生能对着代码看懂每一个像素是怎么来的。
"""

from __future__ import annotations

import math


def _polyline(points: list[tuple[float, float]], stroke: str,
              width: float = 1.0, opacity: float = 1.0) -> str:
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return (f'<polyline fill="none" stroke="{stroke}" stroke-width="{width}"'
            f' opacity="{opacity}" points="{pts}"/>')


def waveform(samples: list[float], width: int = 900, height: int = 160,
             color: str = "#3b82f6") -> str:
    """时域波形。逐样本画折线 —— 16k 样本 900 像素，先按列取
    min/max 包络（保留峰值信息，直接抽样会漏掉瞬态）。"""
    if not samples:
        return "<svg></svg>"
    n = len(samples)
    # 每个像素列的样本区间，取 min/max 两条包络线
    per = n / width
    hi_pts, lo_pts = [], []
    for px in range(width):
        i0, i1 = int(px * per), max(int((px + 1) * per), int(px * per) + 1)
        seg = samples[i0:i1]
        if not seg:
            continue
        hi_pts.append((px, max(seg)))
        lo_pts.append((px, min(seg)))
    mid = height / 2
    scale = mid * 0.92
    top = _polyline([(x, mid - v * scale) for x, v in hi_pts], color)
    bot = _polyline([(x, mid - v * scale) for x, v in lo_pts], color)
    axis = (f'<line x1="0" y1="{mid}" x2="{width}" y2="{mid}" '
            f'stroke="#555" stroke-width="0.5"/>')
    return (f'<svg viewBox="0 0 {width} {height}" xmlns='
            f'"http://www.w3.org/2000/svg" role="img">{axis}{top}{bot}</svg>')


def spectrogram_svg(spec_db: list[list[float]], sample_rate: int,
                    fft_len: int, hop_ms: float,
                    width: int = 900, height: int = 260,
                    floor_db: float = -90.0) -> str:
    """语谱图：x=时间帧，y=频率（线性轴），色深=功率 dB。

    每个像素列对应一帧（帧数多于列宽时按列聚合取最大），
    每个像素行对应一个频率 bin。颜色用黑→蓝→青→黄→白的近似
    viridis 渐变（手写 5 段插值），保持无依赖。
    """
    if not spec_db:
        return "<svg></svg>"
    n_frames = len(spec_db)
    n_bins = len(spec_db[0])
    nyq = sample_rate / 2

    # 频率轴取 Nyquist 的前 0.8（高频通常是空的，压缩掉省像素）
    bins_shown = max(1, int(n_bins * 0.8))

    rects = []
    col_w = width / n_frames
    row_h = height / bins_shown
    for fi, frame in enumerate(spec_db):
        x = fi * col_w
        for bi in range(bins_shown):
            db = frame[bi]
            t = max(0.0, min(1.0, (db - floor_db) / -floor_db))
            if t < 0.02:
                continue                     # 接近 -floor 的底噪不上色
            color = _viridis(t)
            y = height - (bi + 1) * row_h
            rects.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{col_w + 0.4:.2f}" '
                f'height="{row_h + 0.2:.2f}" fill="{color}"/>')

    # 频率刻度：0 / 2k / 4k / Nyquist
    ticks = []
    for f in (0, 2000, 4000):
        if f < nyq * 0.8:
            bi = int(f / nyq * n_bins)
            y = height - bi * row_h
            ticks.append(f'<text x="6" y="{y - 2:.0f}" fill="#888" '
                         f'font-size="10">{f}Hz</text>')
    axis = (f'<line x1="0" y1="{height}" x2="{width}" y2="{height}" '
            f'stroke="#555" stroke-width="0.5"/>')
    return (f'<svg viewBox="0 0 {width} {height}" xmlns='
            f'"http://www.w3.org/2000/svg">{axis}'
            f'{"".join(rects)}{"".join(ticks)}</svg>')


def _viridis(t: float) -> str:
    """黑→蓝→青→黄→白 5 段渐变。t ∈ [0,1]。"""
    stops = [(0, (10, 10, 30)), (0.3, (40, 70, 160)),
             (0.6, (30, 180, 170)), (0.85, (230, 210, 60)),
             (1.0, (255, 255, 240))]
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t <= t1:
            k = (t - t0) / (t1 - t0)
            r = int(c0[0] + (c1[0] - c0[0]) * k)
            g = int(c0[1] + (c1[1] - c0[1]) * k)
            b = int(c0[2] + (c1[2] - c0[2]) * k)
            return f"rgb({r},{g},{b})"
    return "rgb(255,255,240)"


def contour(values: list[float], width: int = 900, height: int = 110,
            color: str = "#f59e0b", zero_line: bool = False,
            unit_suffix: str = "") -> str:
    """逐帧数值的轮廓线（基频/能量/质心通用）。

    值为 0 的帧（未浊音）画灰色点而不是折线断点 —— 断线在
    「这个位置没有基频」和「渲染漏了」之间有歧义，灰点无歧义。
    """
    if not values:
        return "<svg></svg>"
    vmax = max(values) or 1.0
    n = len(values)
    pts = []
    dots = []
    for i, v in enumerate(values):
        x = i * width / max(1, n - 1)
        y = height - (v / vmax) * (height - 8) - 4
        if v > 0:
            pts.append((x, y))
        else:
            dots.append(f'<circle cx="{x:.1f}" cy="{height - 4}" r="1.5" '
                        f'fill="#666"/>')
    parts = [_polyline(pts, color, 1.4)] if pts else []
    parts += dots
    if zero_line:
        parts.append(f'<line x1="0" y1="{height - 4}" x2="{width}" '
                     f'y2="{height - 4}" stroke="#555" '
                     f'stroke-width="0.5" stroke-dasharray="3,3"/>')
    label = (f'<text x="6" y="12" fill="{color}" font-size="11">'
             f'max={vmax:.1f}{unit_suffix}</text>') if unit_suffix else ""
    return (f'<svg viewBox="0 0 {width} {height}" xmlns='
            f'"http://www.w3.org/2000/svg">{"".join(parts)}{label}</svg>')

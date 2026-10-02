# speech-lab — 纯标准库语音信号处理实验台

自实现 radix-2 FFT、自相关基频追踪、MFCC 的语音分析台。
**核心（dsp / features / wavio / synth / visual）只用 Python 标准库**，
FastAPI 只在 HTTP 层 —— 每一个像素都能对着代码讲出来。

```
合成信号（声源-滤波器模型，参数即音频的"源代码"）
   │  16-bit PCM WAV
   ▼
wavio ──► 预加重 ──► 分帧（汉明窗）──► 逐帧特征
   ├─ 基频：自相关法（120Hz 合成 → 119.7Hz 还原）
   ├─ MFCC：Mel 三角滤波器组 → log → DCT-II（13 维）
   ├─ 短时能量 / 过零率 / 频谱质心 / 能量 VAD
   └─ 语谱图：FFT 功率谱 → 相对峰值 60dB 动态范围 → canvas 逐格上色
   ▼
FastAPI（JSON 特征 + SVG 轮廓 + 语谱图网格）──► 单文件前端
```

## 设计决定（都在代码注释里展开了）

- **零依赖核心**：`np.fft.rfft` 一行会黑掉最值得学的 FFT 内部；
  `scripts/check_core_isolation.py` 是 CI 门禁，静态 AST + 动态
  import 双检查，核心拉进任何第三方直接红。
- **合成而非附带 wav**：合成参数就是音频的源代码，可重现、可断言 ——
  测试直接断言"基频追踪还原 120Hz±5"，这是检测算法正确性的最硬证据。
- **语谱图不用 SVG**：98 帧 × 250 bin 的热图 = 2 万多个 rect（1.5MB），
  浏览器直接卡。改为下采样网格 + 前端 canvas 逐格上色。
- **色标相对峰值归一**：FFT 功率谱没有绝对标定，用固定 dB 底线
  会让合成/录音之间大面积饱和或全黑；60dB 相对动态范围铺满 colormap。
- **已知局限写进测试**：自相关基频对超范围高频会报次谐波
  （800Hz → 400Hz），测试把这个行为钉死，让局限可见而不是被掩盖。

## 内置信号（每个都是一类特征的考题）

| 信号 | 考什么 |
|---|---|
| `vowel_a` | 声源-滤波器合成 /a/（F1=730, F2=1090, F3=2440Hz），基频还原 |
| `vowel_burst` | 元音-静音-元音，能量 VAD 必须找到中间静音（voiced≈2/3） |
| `fricative_vowel` | 前半白噪声后半浊音，过零率对比 |
| `two_tones` | 440→880Hz，频谱质心比 ≈ 2 |
| `chirp` | 线性扫频，语谱图应是从左下到右上的斜线 |

也支持上传自己的 16-bit PCM WAV（≤10MB，多声道自动平均）。

## 快速开始

```bash
pip install -r requirements.txt        # fastapi / uvicorn / pydantic
python -m uvicorn app.main:app --port 8134
# 打开 http://127.0.0.1:8134，选信号 → 合成并分析
```

页面上改帧长/帧移再分析，观察语谱图变化 —— 时频不确定性原理
在 DSP 里的样子：帧长越长频率分辨率越高、时间分辨率越低。

## 测试与验收

```bash
python -m pytest -q                      # 57 个单元/集成测试
python scripts/check_core_isolation.py   # 核心零依赖门禁
python scripts/e2e_check.py http://127.0.0.1:8134   # 27 项端到端黑盒
node scripts/shot_frontend.js http://127.0.0.1:8134 docs/screenshots
                                         # 20 项真实浏览器渲染验证 + 6 张截图
```

数值可信度锚点（test_dsp.py）：
- FFT 过 Parseval 定理、与朴素 DFT 定义对拍；
- 合成 120Hz → 基频追踪 119.7Hz；
- 双正弦 440/880Hz → 质心比 2.0（±15%）。

## 目录

```
app/
  dsp.py       ★ FFT/预加重/分帧/窗/能量/ZCR/自相关基频/质心
  features.py  ★ Mel 滤波器组/DCT/MFCC/VAD/FrameAnalysis
  wavio.py     16-bit PCM 读写（wave+struct）/线性重采样/dBFS
  synth.py     声源-滤波器合成器（Klatt 谐振器差分方程）
  visual.py    SVG 波形/轮廓渲染
  main.py      FastAPI（analyze / analyze/upload / signals）
web/index.html 单文件前端（canvas 语谱图 + SVG 特征轮廓）
scripts/       e2e、零依赖门禁、浏览器渲染验证
tests/         57 个测试（dsp 34 含数值锚点 / features / wavio / e2e 特征）
```

MIT License

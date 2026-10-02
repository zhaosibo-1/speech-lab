"""零依赖门禁：app/ 的核心 DSP 模块只许用 Python 标准库。

这是 speech-lab 的立身之本 —— 「每一个像素都能对着代码讲出来」
的前提是核心没有黑盒。fastapi/pydantic 只允许出现在 main.py。

两层检查：
1. 静态：AST 扫描所有 import，对照标准库白名单；
2. 动态：真正 import 一遍核心模块后查 sys.modules，
   静态扫描漏掉的动态导入（或传递依赖）会在这里现形。
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "app"
CORE = {"dsp", "features", "wavio", "synth", "visual"}   # main.py 豁免

# 标准库白名单（按需维护；import stdlib list 太严会误伤 __future__ 等）
STDLIB = set(sys.stdlib_module_names) | {"__future__"}


def static_check() -> list[str]:
    problems = []
    for py in sorted(APP.glob("*.py")):
        module = py.stem
        if module not in CORE:
            continue
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module or ""]
            for name in names:
                root = name.split(".")[0]
                if root not in STDLIB:
                    problems.append(
                        f"{py.name}:{node.lineno} import {name} "
                        f"—— 核心模块只允许标准库")
    return problems


def dynamic_check() -> list[str]:
    """在干净的解释器里 import 核心包，看有没有拉进第三方。"""
    import subprocess
    code = (
        "import sys; sys.path.insert(0, r'%s');\n"
        "import app.dsp, app.features, app.wavio, app.synth, app.visual;\n"
        "third = sorted(m for m in sys.modules\n"
        "    if not m.startswith('_') and m.split('.')[0] not in\n"
        "    set(sys.stdlib_module_names) | {'app', 'sitecustomize',"
        " 'usercustomize'} and '.' not in m);\n"
        "print('\\n'.join(third))\n" % str(APP.parent)
    )
    proc = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True)
    third = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    return [f"动态检查：核心 import 后出现第三方模块 {m}" for m in third]


def main() -> int:
    problems = static_check() + dynamic_check()
    if problems:
        print("✗ 零依赖门禁未通过：")
        for p in problems:
            print("  " + p)
        return 1
    print(f"✓ 通过：{len(CORE)} 个核心模块零第三方依赖（静态 AST + 动态 import 双检查）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

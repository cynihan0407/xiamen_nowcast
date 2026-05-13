"""pytest 配置：让测试无需 ``pip install -e`` 也可 import ``src``。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 尽早发现「未提交 / 未拉取」的 ``src/data`` 包（常见于 .gitignore 误伤后服务器缺文件）
try:
    import src.data.normalizers  # noqa: F401
    import src.data.h8_dataset  # noqa: F401
except ImportError as e:
    data_dir = ROOT / "src" / "data"
    raise ImportError(
        "无法导入 ``src.data``：请确认仓库中存在 Python 包 ``src/data/``（含 "
        "``normalizers.py``、``h8_dataset.py`` 等）。\n"
        "若本地曾使用规则 ``data/`` 的 ``.gitignore``，会误忽略 ``src/data/``，"
        "导致 push 后服务器缺源码；请拉取最新 ``.gitignore`` 后执行：\n"
        "  git add src/data && git commit -m \"fix: track src/data package\"\n"
        f"当前检查路径: {data_dir}  exists={data_dir.is_dir()}"
    ) from e

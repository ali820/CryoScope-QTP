from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cryoscope_core as core


REQUIRED_PATHS = [
    ROOT_DIR / "cryoscope_core.py",
    ROOT_DIR / "cryoscope_qtp_streamlit_prototype_v2.py",
    ROOT_DIR / "README.md",
    ROOT_DIR / "docs" / "v1_scope.md",
    ROOT_DIR / "docs" / "release_checklist.md",
    ROOT_DIR / "tests" / "test_core.py",
    ROOT_DIR / "rules",
    ROOT_DIR / "knowledge",
]


def ensure_paths() -> None:
    missing = [path for path in REQUIRED_PATHS if not path.exists()]
    if missing:
        raise RuntimeError("缺少关键项目文件: " + ", ".join(str(path.relative_to(ROOT_DIR)) for path in missing))


def ensure_core_logic() -> None:
    prompt = core.build_fast_compiled_prompt(
        user_prompt="当前湖岸是否存在退化风险",
        task_mode="risk",
        research_context="青藏高原湖岸样区。",
        files=[],
        knowledge_hits=[
            {
                "source": "knowledge/mechanism/permafrost_basics.md",
                "chunk_id": 0,
                "text": "需要区分观测事实与机理推断。",
            }
        ],
    )
    if "本地知识命中" not in prompt:
        raise RuntimeError("快速提示词未包含知识命中摘要。")

    df = pd.DataFrame({"temp": [1.0, 2.0, 3.0], "deformation": [0.1, 0.2, 0.3]})
    analysis = core.analyze_dataframe(df)
    if "数据规模：3 行，2 列。" not in analysis:
        raise RuntimeError("数据分析摘要异常。")

    result = core.diagnose_uav_dem(
        {
            "rtk_first": "无",
            "rtk_second": "有",
            "same_vertical_datum": "否",
            "gcp_quality": "低",
            "strip_shape": "不规则长条航带",
            "alignment_method": "人工粗配准",
            "vegetation": "高",
            "water_edge": True,
            "shadow_occlusion": True,
            "stable_zone_checked": False,
            "max_change": 0.10,
        }
    )
    if result["level"] != "高":
        raise RuntimeError("航测/DEM 风险诊断结果异常。")


def main() -> int:
    ensure_paths()
    ensure_core_logic()
    print("Smoke check passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Smoke check failed: {exc}", file=sys.stderr)
        raise SystemExit(1)

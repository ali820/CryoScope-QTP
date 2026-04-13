import unittest

import pandas as pd

import cryoscope_core as core


class FakeUpload:
    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


class CoreLogicTests(unittest.TestCase):
    def test_resolve_llm_settings_defaults_to_kimi_when_kimi_key_is_present(self) -> None:
        settings = core.resolve_llm_settings(
            secret_reader=lambda _: None,
            environ={"KIMI_API_KEY": "kimi-key"},
        )
        self.assertEqual(settings.provider, "kimi")
        self.assertEqual(settings.api_key, "kimi-key")
        self.assertEqual(settings.base_url, core.KIMI_DEFAULT_BASE_URL)
        self.assertEqual(settings.model, core.KIMI_DEFAULT_MODEL)
        self.assertEqual(settings.fallback_models, [core.KIMI_DEFAULT_MODEL])

    def test_resolve_llm_settings_accepts_moonshot_aliases(self) -> None:
        settings = core.resolve_llm_settings(
            secret_reader=lambda _: None,
            environ={
                "MOONSHOT_API_KEY": "moonshot-key",
                "MOONSHOT_MODEL": "moonshot-v1-32k",
                "MOONSHOT_FALLBACK_MODELS": "moonshot-v1-8k, moonshot-v1-32k",
            },
        )
        self.assertEqual(settings.provider, "kimi")
        self.assertEqual(settings.api_key, "moonshot-key")
        self.assertEqual(settings.base_url, core.KIMI_DEFAULT_BASE_URL)
        self.assertEqual(settings.model, "moonshot-v1-32k")
        self.assertEqual(settings.fallback_models, ["moonshot-v1-8k", "moonshot-v1-32k"])

    def test_resolve_llm_settings_detects_kimi_from_openai_compatible_base_url(self) -> None:
        settings = core.resolve_llm_settings(
            secret_reader=lambda _: None,
            environ={
                "OPENAI_API_KEY": "generic-key",
                "OPENAI_BASE_URL": core.KIMI_DEFAULT_BASE_URL,
            },
        )
        self.assertEqual(settings.provider, "kimi")
        self.assertEqual(settings.model, core.KIMI_DEFAULT_MODEL)
        self.assertEqual(settings.fallback_models, [core.KIMI_DEFAULT_MODEL])

    def test_build_fast_compiled_prompt_keeps_knowledge_hits(self) -> None:
        prompt = core.build_fast_compiled_prompt(
            user_prompt="湖岸后退是不是冻土退化",
            task_mode="risk",
            research_context="青藏高原湖岸，已有 InSAR 与两期 UAV。",
            files=[],
            knowledge_hits=[
                {
                    "source": "knowledge/mechanism/permafrost_basics.md",
                    "chunk_id": 0,
                    "text": "判断退化时需要区分观测、推断和不确定性。",
                }
            ],
        )
        self.assertIn("本地知识命中", prompt)
        self.assertIn("permafrost_basics.md", prompt)

    def test_extract_uploaded_file_returns_error_instead_of_raising(self) -> None:
        uploaded = FakeUpload("broken.xlsx", b"not-a-real-excel-file")
        extracted = core.extract_uploaded_file(uploaded)
        self.assertEqual(extracted.kind, "error")
        self.assertIn("文件解析失败", extracted.text)

    def test_analyze_dataframe_summarizes_numeric_columns(self) -> None:
        df = pd.DataFrame(
            {
                "ground_temp": [1.0, 2.0, 3.0, 4.0],
                "deformation": [0.1, 0.2, 0.3, 0.4],
            }
        )
        analysis = core.analyze_dataframe(df)
        self.assertIn("数据规模：4 行，2 列。", analysis)
        self.assertIn("相关性较强的字段对：", analysis)
        self.assertIn("ground_temp vs deformation", analysis)

    def test_diagnose_uav_dem_flags_high_risk_case(self) -> None:
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
        self.assertEqual(result["level"], "高")
        self.assertTrue(any("垂向基准" in item for item in result["findings"]))
        self.assertTrue(any("误差" in item for item in result["suggestions"]))


if __name__ == "__main__":
    unittest.main()

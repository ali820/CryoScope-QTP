from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document
except ImportError:
    Document = None


APP_NAME = "CryoScope-QTP"
APP_SUBTITLE = "青藏高原冻土多源分析与科研辅助 Agent"
ROOT_DIR = Path(__file__).resolve().parent
RULES_DIR = ROOT_DIR / "rules"
KNOWLEDGE_DIR = ROOT_DIR / "knowledge"
SUPPORTED_TEXT_SUFFIXES = {".md", ".txt"}
SUPPORTED_TABLE_SUFFIXES = {".csv", ".xlsx", ".xls"}
SUPPORTED_DOC_SUFFIXES = {".pdf", ".docx"}
CUSTOM_GATEWAY_PROMPT_LIMIT = 2800
DEFAULT_CUSTOM_GATEWAY_FALLBACKS = [
    "gpt-oss-120b",
    "deepseek-r1:32b",
    "deepseek-v3:671b",
]
OPENAI_DEFAULT_MODEL = "gpt-4.1-mini"
KIMI_DEFAULT_BASE_URL = "https://api.moonshot.cn/v1"
KIMI_DEFAULT_MODEL = "moonshot-v1-8k"
KIMI_DEFAULT_FALLBACKS = [KIMI_DEFAULT_MODEL]
OPENAI_API_KEY_NAMES = ("OPENAI_API_KEY", "KIMI_API_KEY", "MOONSHOT_API_KEY")
OPENAI_BASE_URL_NAMES = ("OPENAI_BASE_URL", "KIMI_BASE_URL", "MOONSHOT_BASE_URL")
OPENAI_MODEL_NAMES = ("OPENAI_MODEL", "KIMI_MODEL", "MOONSHOT_MODEL")
OPENAI_FALLBACK_MODEL_NAMES = (
    "OPENAI_FALLBACK_MODELS",
    "KIMI_FALLBACK_MODELS",
    "MOONSHOT_FALLBACK_MODELS",
)
KIMI_CONFIG_NAMES = {
    "KIMI_API_KEY",
    "MOONSHOT_API_KEY",
    "KIMI_BASE_URL",
    "MOONSHOT_BASE_URL",
    "KIMI_MODEL",
    "MOONSHOT_MODEL",
    "KIMI_FALLBACK_MODELS",
    "MOONSHOT_FALLBACK_MODELS",
}

TASK_MODE_CONFIG = {
    "concept": {
        "label": "概念解释",
        "rule_file": "rules_permafrost.md",
        "focus": "解释多年冻土、活动层、热喀斯特、冻胀融沉、水热过程等概念，并给出概念边界。",
    },
    "literature": {
        "label": "文献归纳",
        "rule_file": "rules_permafrost.md",
        "focus": "围绕冻土研究问题梳理研究进展、常见方法、变量体系、证据链和研究空白。",
    },
    "data": {
        "label": "数据解读",
        "rule_file": "rules_insar.md",
        "focus": "结合时序、表格、地温、降水、形变等数据做特征提取、异常识别和机理解释。",
    },
    "uav_dem": {
        "label": "航测/DEM 诊断",
        "rule_file": "rules_uav_dem.md",
        "focus": "判断两期 UAV/DEM 差分、配准和高程基准是否可靠，并列出潜在误差来源。",
    },
    "writing": {
        "label": "学术写作",
        "rule_file": "rules_writing.md",
        "focus": "生成摘要、方法、讨论、审稿回复、基金申请等学术文本，保持克制和可核查。",
    },
    "risk": {
        "label": "风险判断",
        "rule_file": "rules_permafrost.md",
        "focus": "面向湖岸、渠道、路基、边坡等工程/地貌场景输出风险等级、触发因素和监测建议。",
    },
}

BASE_SYSTEM_PROMPT = """
你是 CryoScope-QTP，一名面向青藏高原多年冻土研究的专业智能助手。

你的目标不是泛泛聊天，而是围绕以下场景提供帮助：
1. 冻土概念解释与机理辨析
2. 文献整理与研究设计
3. InSAR、UAV、DEM/DOM、地温、气象、湖位等多源数据解读
4. 航测/DEM 差分质量诊断
5. 学术写作与审稿回复
6. 冻土工程与湖岸退化风险判断

必须遵守以下原则：
1. 区分“观测事实”“推断解释”“仍不确定的部分”。
2. 不把 InSAR 形变、DEM 差分或局部地貌变化直接等同于冻土退化结论。
3. 当信息不足时，明确提出还需要什么数据，而不是强行下结论。
4. 优先输出结构化内容，避免空泛结论。
5. 如果用户上传了文件或给出本地知识片段，优先基于这些材料作答。
6. 除写作模式外，默认按“结论 / 依据 / 替代解释 / 不确定性 / 建议”组织回答。
7. 如果需要做学术写作，不得编造文献、数据或实验结果。
""".strip()


@dataclass
class ExtractedFile:
    name: str
    suffix: str
    kind: str
    text: str
    preview: str


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    api_key: str | None
    base_url: str | None
    model: str
    fallback_models: list[str]


def resolve_config_value(
    names: tuple[str, ...],
    secret_reader: Callable[[str], str | None],
    environ: Mapping[str, str] | None = None,
) -> tuple[str | None, str | None]:
    source_env = environ if environ is not None else os.environ
    for name in names:
        secret_value = secret_reader(name)
        if secret_value:
            return secret_value, name
    for name in names:
        env_value = source_env.get(name)
        if env_value:
            return env_value, name
    return None, None


def parse_fallback_models(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def infer_llm_provider(
    base_url: str | None,
    model: str | None,
    explicit_sources: tuple[str | None, ...] = (),
) -> str:
    if any(source in KIMI_CONFIG_NAMES for source in explicit_sources if source):
        return "kimi"
    normalized_base_url = (base_url or "").lower()
    normalized_model = (model or "").lower()
    if "moonshot.cn" in normalized_base_url:
        return "kimi"
    if normalized_model.startswith("moonshot-") or normalized_model.startswith("kimi-"):
        return "kimi"
    return "openai"


def get_default_fallback_models(provider: str, base_url: str | None) -> list[str]:
    if provider == "kimi":
        return KIMI_DEFAULT_FALLBACKS.copy()
    if is_custom_gateway(base_url):
        return DEFAULT_CUSTOM_GATEWAY_FALLBACKS.copy()
    return []


def resolve_llm_settings(
    secret_reader: Callable[[str], str | None],
    environ: Mapping[str, str] | None = None,
) -> LLMSettings:
    api_key, api_key_source = resolve_config_value(
        OPENAI_API_KEY_NAMES,
        secret_reader,
        environ,
    )
    base_url, base_url_source = resolve_config_value(
        OPENAI_BASE_URL_NAMES,
        secret_reader,
        environ,
    )
    model, model_source = resolve_config_value(
        OPENAI_MODEL_NAMES,
        secret_reader,
        environ,
    )
    fallback_raw, fallback_source = resolve_config_value(
        OPENAI_FALLBACK_MODEL_NAMES,
        secret_reader,
        environ,
    )

    provider = infer_llm_provider(
        base_url=base_url,
        model=model,
        explicit_sources=(api_key_source, base_url_source, model_source, fallback_source),
    )

    if provider == "kimi":
        resolved_base_url = base_url or KIMI_DEFAULT_BASE_URL
        resolved_model = model or KIMI_DEFAULT_MODEL
    else:
        resolved_base_url = base_url
        resolved_model = model or OPENAI_DEFAULT_MODEL

    fallback_models = parse_fallback_models(fallback_raw)
    if not fallback_models:
        fallback_models = get_default_fallback_models(provider, resolved_base_url)

    return LLMSettings(
        provider=provider,
        api_key=api_key,
        base_url=resolved_base_url,
        model=resolved_model,
        fallback_models=fallback_models,
    )


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def trim_text(text: str, limit: int = 3000) -> str:
    cleaned = text.strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "\n...[已截断]"


def safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def list_rule_files() -> list[Path]:
    if not RULES_DIR.exists():
        return []
    return sorted(path for path in RULES_DIR.glob("*.md") if path.is_file())


def load_rule_bundle(task_mode: str) -> str:
    config = TASK_MODE_CONFIG[task_mode]
    bundle = []
    shared = RULES_DIR / "rules.md"
    if shared.exists():
        bundle.append(f"# {shared.name}\n{safe_read_text(shared)}")
    specific = RULES_DIR / config["rule_file"]
    if specific.exists():
        bundle.append(f"# {specific.name}\n{safe_read_text(specific)}")
    return "\n\n".join(bundle).strip()


def build_system_prompt(task_mode: str) -> str:
    config = TASK_MODE_CONFIG[task_mode]
    rule_text = load_rule_bundle(task_mode)
    output_hint = (
        "当前模式优先输出：任务理解 / 草稿正文 / 可替换变量 / 修改建议。"
        if task_mode == "writing"
        else "当前模式优先输出：结论 / 依据 / 替代解释 / 不确定性 / 建议。"
    )
    return (
        f"{BASE_SYSTEM_PROMPT}\n\n"
        f"当前任务模式：{config['label']}\n"
        f"当前模式目标：{config['focus']}\n"
        f"{output_hint}\n\n"
        f"必须遵守的领域规则：\n{rule_text if rule_text else '当前没有额外规则文件。'}"
    )


def build_fast_system_prompt(task_mode: str) -> str:
    config = TASK_MODE_CONFIG[task_mode]
    return (
        "你是 CryoScope-QTP，面向青藏高原冻土研究。\n"
        "请简洁、谨慎作答，不把推断说成证实；信息不足时直接说缺什么。\n"
        f"当前任务模式：{config['label']}\n"
        "默认按“结论 / 依据 / 不确定性 / 建议”作答。"
    )


def extract_text_from_pdf_bytes(data: bytes) -> str:
    if PdfReader is None:
        return "未安装 pypdf，当前无法提取 PDF 正文。"
    try:
        reader = PdfReader(io.BytesIO(data))
        parts = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        text = "\n".join(parts).strip()
        return text or "PDF 未提取到可读正文，可能是扫描版或图片版。"
    except Exception as exc:
        return f"PDF 解析失败：{exc}"


def extract_text_from_docx_bytes(data: bytes) -> str:
    if Document is None:
        return "未安装 python-docx，当前无法提取 DOCX 正文。"
    try:
        doc = Document(io.BytesIO(data))
        lines = [paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()]
        text = "\n".join(lines).strip()
        return text or "DOCX 中未提取到正文。"
    except Exception as exc:
        return f"DOCX 解析失败：{exc}"


def load_dataframe_from_bytes(name: str, data: bytes) -> pd.DataFrame:
    suffix = Path(name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(io.BytesIO(data))
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(io.BytesIO(data))
    raise ValueError(f"暂不支持的数据表格式：{suffix}")


def summarize_dataframe(df: pd.DataFrame) -> tuple[str, str]:
    row_count, col_count = df.shape
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    missing_counts = df.isna().sum()
    top_missing = missing_counts[missing_counts > 0].sort_values(ascending=False).head(8)
    summary_lines = [
        f"行数：{row_count}",
        f"列数：{col_count}",
        f"字段：{', '.join(map(str, df.columns.tolist()))}",
        f"数值字段：{', '.join(numeric_cols) if numeric_cols else '无'}",
    ]
    if not top_missing.empty:
        summary_lines.append(
            "缺失值较多字段："
            + ", ".join(f"{name}={int(value)}" for name, value in top_missing.items())
        )
    if numeric_cols:
        stats = (
            df[numeric_cols]
            .describe()
            .T[["mean", "std", "min", "max"]]
            .round(4)
            .reset_index()
            .rename(columns={"index": "field"})
        )
        summary_lines.append("数值字段统计：\n" + stats.to_csv(index=False))
    preview = df.head(12).to_csv(index=False)
    return "\n".join(summary_lines), preview


def extract_uploaded_file(uploaded_file: Any) -> ExtractedFile:
    suffix = Path(uploaded_file.name).suffix.lower()
    data = uploaded_file.getvalue()
    try:
        if suffix in SUPPORTED_TEXT_SUFFIXES:
            text = data.decode("utf-8", errors="ignore")
            return ExtractedFile(
                name=uploaded_file.name,
                suffix=suffix,
                kind="text",
                text=text,
                preview=trim_text(text, 1500),
            )
        if suffix in SUPPORTED_TABLE_SUFFIXES:
            df = load_dataframe_from_bytes(uploaded_file.name, data)
            summary, preview = summarize_dataframe(df)
            combined = f"{summary}\n\n样例数据：\n{preview}"
            return ExtractedFile(
                name=uploaded_file.name,
                suffix=suffix,
                kind="table",
                text=combined,
                preview=trim_text(combined, 1800),
            )
        if suffix == ".pdf":
            text = extract_text_from_pdf_bytes(data)
            return ExtractedFile(
                name=uploaded_file.name,
                suffix=suffix,
                kind="pdf",
                text=text,
                preview=trim_text(text, 1800),
            )
        if suffix == ".docx":
            text = extract_text_from_docx_bytes(data)
            return ExtractedFile(
                name=uploaded_file.name,
                suffix=suffix,
                kind="docx",
                text=text,
                preview=trim_text(text, 1800),
            )
    except Exception as exc:
        message = f"文件解析失败：{exc}"
        return ExtractedFile(
            name=uploaded_file.name,
            suffix=suffix,
            kind="error",
            text=message,
            preview=message,
        )
    return ExtractedFile(
        name=uploaded_file.name,
        suffix=suffix,
        kind="unsupported",
        text="当前文件类型尚未纳入自动解析。",
        preview="当前文件类型尚未纳入自动解析。",
    )


def extract_text_from_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in SUPPORTED_TEXT_SUFFIXES:
        return safe_read_text(path)
    if suffix == ".pdf":
        return extract_text_from_pdf_bytes(path.read_bytes())
    if suffix == ".docx":
        return extract_text_from_docx_bytes(path.read_bytes())
    if suffix in SUPPORTED_TABLE_SUFFIXES:
        df = load_dataframe_from_bytes(path.name, path.read_bytes())
        summary, preview = summarize_dataframe(df)
        return f"{summary}\n\n样例数据：\n{preview}"
    return ""


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> list[str]:
    cleaned = normalize_text(text)
    if not cleaned:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)
        chunks.append(cleaned[start:end])
        if end >= len(cleaned):
            break
        start = max(0, end - overlap)
    return chunks


def tokenize_text(text: str) -> list[str]:
    lowered = text.lower()
    tokens = re.findall(r"[a-z0-9_+-]{2,}", lowered)
    han_blocks = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    for block in han_blocks:
        tokens.append(block)
        if len(block) <= 12:
            tokens.extend(block[idx : idx + 2] for idx in range(len(block) - 1))
            if len(block) >= 4:
                tokens.extend(block[idx : idx + 3] for idx in range(len(block) - 2))
    return list(dict.fromkeys(token for token in tokens if len(token) >= 2))


def build_knowledge_signature(root: Path) -> str:
    if not root.exists():
        return "missing"
    parts = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            stat = path.stat()
            parts.append(f"{path.relative_to(root)}:{stat.st_mtime_ns}:{stat.st_size}")
    return "|".join(parts)


def build_knowledge_entries(root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if not root.exists():
        return entries
    supported = SUPPORTED_TEXT_SUFFIXES | SUPPORTED_TABLE_SUFFIXES | SUPPORTED_DOC_SUFFIXES
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in supported:
            continue
        text = extract_text_from_path(path)
        for chunk_id, chunk in enumerate(chunk_text(text)):
            entries.append(
                {
                    "source": str(path.relative_to(ROOT_DIR)),
                    "chunk_id": chunk_id,
                    "text": chunk,
                    "tokens": tokenize_text(chunk),
                }
            )
    return entries


def score_entry(query: str, query_tokens: list[str], entry: dict[str, Any]) -> float:
    if not query_tokens:
        return 0.0
    token_set = set(entry["tokens"])
    overlap = {token for token in query_tokens if token in token_set}
    if not overlap:
        return 0.0
    score = float(len(overlap) * 3)
    lowered_text = entry["text"].lower()
    lowered_query = normalize_text(query).lower()
    for token in overlap:
        score += lowered_text.count(token.lower())
    if lowered_query and lowered_query in lowered_text:
        score += 5.0
    for token in overlap:
        if token.lower() in entry["source"].lower():
            score += 1.5
    return score


def retrieve_knowledge_from_entries(
    query: str,
    entries: list[dict[str, Any]],
    top_k: int = 4,
) -> list[dict[str, Any]]:
    query_tokens = tokenize_text(query)
    scored = []
    for entry in entries:
        score = score_entry(query, query_tokens, entry)
        if score > 0:
            scored.append((score, entry))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [entry for _, entry in scored[:top_k]]


def format_file_context(files: list[ExtractedFile]) -> str:
    if not files:
        return "无上传文件。"
    blocks = []
    for item in files:
        blocks.append(
            f"文件名：{item.name}\n"
            f"类型：{item.kind}\n"
            f"可用摘录：\n{trim_text(item.text, 2400)}"
        )
    return "\n\n".join(blocks)


def format_knowledge_context(hits: list[dict[str, Any]]) -> str:
    if not hits:
        return "未命中本地知识库。"
    blocks = []
    for index, hit in enumerate(hits, start=1):
        blocks.append(
            f"[知识片段 {index}] 来源：{hit['source']}#chunk-{hit['chunk_id']}\n"
            f"{trim_text(hit['text'], 900)}"
        )
    return "\n\n".join(blocks)


def is_custom_gateway(base_url: str | None) -> bool:
    if not base_url:
        return False
    return "api.openai.com" not in base_url


def build_compiled_prompt(
    user_prompt: str,
    task_mode: str,
    research_context: str,
    files: list[ExtractedFile],
    knowledge_hits: list[dict[str, Any]],
) -> str:
    response_hint = (
        "请输出：任务理解、草稿正文、可替换变量、修改建议。"
        if task_mode == "writing"
        else "请优先输出：结论、依据、替代解释、不确定性、建议。"
    )
    return (
        f"用户问题：\n{user_prompt}\n\n"
        f"任务模式：{TASK_MODE_CONFIG[task_mode]['label']}\n"
        f"研究背景：\n{research_context or '未提供'}\n\n"
        f"本地知识库命中：\n{format_knowledge_context(knowledge_hits)}\n\n"
        f"上传文件上下文：\n{format_file_context(files)}\n\n"
        f"{response_hint}"
    )


def build_fast_compiled_prompt(
    user_prompt: str,
    task_mode: str,
    research_context: str,
    files: list[ExtractedFile],
    knowledge_hits: list[dict[str, Any]],
) -> str:
    compact_files = files[:1]
    compact_knowledge = knowledge_hits[:2]
    parts = [
        f"用户问题：{trim_text(user_prompt, 500)}",
        f"任务模式：{TASK_MODE_CONFIG[task_mode]['label']}",
    ]
    if research_context.strip():
        parts.append(f"研究背景：{trim_text(research_context, 220)}")
    if compact_knowledge:
        parts.append(
            f"本地知识命中：{trim_text(format_knowledge_context(compact_knowledge), 280)}"
        )
    if compact_files:
        parts.append(f"上传材料摘录：{trim_text(format_file_context(compact_files), 350)}")
    parts.append("请尽量直接回答，不要长篇铺垫。")
    return trim_text("\n".join(parts), CUSTOM_GATEWAY_PROMPT_LIMIT)


def build_offline_reply(
    user_prompt: str,
    task_mode: str,
    files: list[ExtractedFile],
    knowledge_hits: list[dict[str, Any]],
) -> str:
    lines = [
        f"当前处于“{TASK_MODE_CONFIG[task_mode]['label']}”模式。",
        f"收到的问题：{user_prompt}",
    ]
    if knowledge_hits:
        lines.append(
            "本地知识库已命中："
            + "；".join(f"{item['source']}#chunk-{item['chunk_id']}" for item in knowledge_hits)
        )
    else:
        lines.append("本地知识库未命中明显相关片段。")
    if files:
        lines.append("已读取上传文件：" + "，".join(item.name for item in files))
    else:
        lines.append("当前没有上传文件。")
    lines.append(
        "未检测到可用 API 配置，当前未调用大模型。配置 `OPENAI_API_KEY` 或 `KIMI_API_KEY` 后，主聊天页会自动切换为真实 agent。"
    )
    return "\n\n".join(lines)


def detect_trend(series: pd.Series) -> str:
    clean = series.dropna().astype(float)
    if len(clean) < 3:
        return "样本过少，趋势不稳定"
    x = np.arange(len(clean))
    slope = float(np.polyfit(x, clean.to_numpy(), 1)[0])
    span = float(clean.max() - clean.min())
    if span == 0:
        return "基本稳定"
    normalized = abs(slope) * len(clean) / span
    if normalized < 0.2:
        return "整体较稳定"
    if slope > 0:
        return "整体上升"
    return "整体下降"


def analyze_dataframe(df: pd.DataFrame) -> str:
    numeric = df.select_dtypes(include="number")
    lines = [
        f"数据规模：{df.shape[0]} 行，{df.shape[1]} 列。",
        "字段列表：" + ", ".join(map(str, df.columns.tolist())),
    ]
    if numeric.empty:
        lines.append("当前表中没有可直接用于趋势分析的数值字段。")
        return "\n".join(lines)

    lines.append("数值字段快照：")
    for column in numeric.columns[:8]:
        series = numeric[column]
        non_null = series.dropna()
        if non_null.empty:
            continue
        mean = float(non_null.mean())
        std = float(non_null.std(ddof=0))
        anomaly_count = 0
        if std > 0:
            anomaly_count = int(((non_null - mean).abs() / std > 3).sum())
        lines.append(
            f"- {column}: 最小值={non_null.min():.4f}, 最大值={non_null.max():.4f}, "
            f"均值={mean:.4f}, 趋势={detect_trend(non_null)}, 异常值候选={anomaly_count}"
        )

    corr = numeric.corr(numeric_only=True)
    if corr.shape[0] >= 2:
        melted = (
            corr.where(~np.eye(corr.shape[0], dtype=bool))
            .stack()
            .sort_values(key=lambda value: value.abs(), ascending=False)
        )
        top_pairs = []
        used = set()
        for (left, right), value in melted.items():
            key = tuple(sorted((left, right)))
            if key in used:
                continue
            used.add(key)
            top_pairs.append((left, right, float(value)))
            if len(top_pairs) == 3:
                break
        if top_pairs:
            lines.append("相关性较强的字段对：")
            for left, right, value in top_pairs:
                lines.append(f"- {left} vs {right}: r={value:.3f}")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False).head(5)
    if not missing.empty:
        lines.append(
            "缺失值提醒：" + ", ".join(f"{name}={int(value)}" for name, value in missing.items())
        )
    return "\n".join(lines)


def diagnose_uav_dem(payload: dict[str, Any]) -> dict[str, Any]:
    score = 0
    findings = []
    suggestions = []
    uncertainties = []

    if payload["rtk_first"] == "无" and payload["rtk_second"] == "有":
        score += 2
        findings.append("两期 RTK 条件不一致，存在垂向基准不统一风险。")
        suggestions.append("优先核查两期高程基准、控制点和解算流程是否一致。")
    if payload["same_vertical_datum"] != "是":
        score += 2
        findings.append("垂向基准未确认一致，DoD 结果可能混入系统偏移。")
        suggestions.append("建立统一垂向基准，或用稳定区做垂向偏移校正。")
    if payload["strip_shape"] == "不规则长条航带":
        score += 2
        findings.append("不规则长条航带容易积累条带误差和边缘漂移。")
        suggestions.append("增加稳定区约束，重点检查条带两端与拼接边界。")
    if payload["gcp_quality"] == "低":
        score += 2
        findings.append("地面控制质量偏低，配准稳定性不足。")
        suggestions.append("补充高质量 GCP 或稳定 RTK 检核点。")
    elif payload["gcp_quality"] == "中":
        score += 1
    if payload["alignment_method"] in {"人工粗配准", "未知"}:
        score += 1
        findings.append("当前配准方式较弱，难以支撑小幅高程变化解释。")
        suggestions.append("优先使用稳定区 + GCP/ICP 的联合配准方案。")
    if payload["water_edge"]:
        score += 1
        findings.append("水边界附近容易出现匹配失败和虚假高程变化。")
        suggestions.append("对水边、浪蚀带和湿润区单独设置信任等级。")
    if payload["vegetation"] in {"中", "高"}:
        score += 1
        findings.append("植被覆盖会放大表面模型差异，不宜直接解释为地表升降。")
        suggestions.append("区分 DSM 与裸地 DEM 的差异，必要时做地物掩膜。")
    if payload["shadow_occlusion"]:
        score += 1
        findings.append("阴影/遮挡区的差分结果可信度较低。")
        suggestions.append("对阴影区做掩膜，避免把遮挡误差解释为侵蚀或沉降。")
    if payload["max_change"] <= 0.15 and score >= 4:
        uncertainties.append("当前观测变化量较小，可能与配准误差同量级。")
    if not payload["stable_zone_checked"]:
        score += 1
        findings.append("尚未明确说明是否用稳定区检核误差。")
        suggestions.append("先在稳定裸地/硬化区评估 DoD 背景噪声，再解释变化热点。")

    if score <= 2:
        level = "低"
    elif score <= 5:
        level = "中"
    else:
        level = "高"

    if not findings:
        findings.append("当前输入未暴露明显的高风险误差项，但仍需检查稳定区残差。")
    if not uncertainties:
        uncertainties.append("缺少原始控制点残差、稳定区误差统计和遮挡掩膜信息。")
    if not suggestions:
        suggestions.append("补充稳定区检核、垂向基准说明和误差统计表。")

    return {
        "level": level,
        "findings": findings,
        "uncertainties": uncertainties,
        "suggestions": suggestions,
    }

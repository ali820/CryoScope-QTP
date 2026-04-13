from __future__ import annotations

import concurrent.futures
import time
from typing import Any

import streamlit as st

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from cryoscope_core import (
    APP_NAME,
    APP_SUBTITLE,
    KNOWLEDGE_DIR,
    ROOT_DIR,
    TASK_MODE_CONFIG,
    analyze_dataframe,
    build_compiled_prompt,
    build_fast_compiled_prompt,
    build_fast_system_prompt,
    build_knowledge_entries,
    build_knowledge_signature,
    build_offline_reply,
    build_system_prompt,
    diagnose_uav_dem,
    extract_uploaded_file,
    get_default_fallback_models,
    infer_llm_provider,
    is_custom_gateway,
    list_rule_files,
    load_dataframe_from_bytes,
    retrieve_knowledge_from_entries,
    resolve_llm_settings,
    safe_read_text,
    trim_text,
)


MODEL_TIMEOUT_SECONDS = 90
CUSTOM_GATEWAY_REQUEST_TIMEOUT_SECONDS = 20.0
OPENAI_REQUEST_TIMEOUT_SECONDS = 60.0


@st.cache_resource(show_spinner=False)
def build_knowledge_index(signature: str) -> list[dict[str, Any]]:
    del signature
    return build_knowledge_entries(KNOWLEDGE_DIR)


def retrieve_knowledge(query: str, top_k: int = 4) -> list[dict[str, Any]]:
    signature = build_knowledge_signature(KNOWLEDGE_DIR)
    entries = build_knowledge_index(signature)
    return retrieve_knowledge_from_entries(query, entries, top_k=top_k)


def get_secret(name: str) -> str | None:
    try:
        value = st.secrets[name]
    except Exception:
        return None
    return str(value) if value is not None else None


def build_api_messages(
    history: list[dict[str, str]],
    system_prompt: str,
    compiled_user_prompt: str,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": [{"type": "input_text", "text": system_prompt}],
        }
    ]
    for item in history[-4:]:
        messages.append(
            {
                "role": item["role"],
                "content": [{"type": "input_text", "text": item["content"]}],
            }
        )
    messages.append(
        {
            "role": "user",
            "content": [{"type": "input_text", "text": compiled_user_prompt}],
        }
    )
    return messages


def call_model(
    api_key: str,
    base_url: str | None,
    model: str,
    fallback_models: list[str],
    system_prompt: str,
    history: list[dict[str, str]],
    compiled_user_prompt: str,
) -> tuple[str, str]:
    if OpenAI is None:
        raise RuntimeError("未安装 openai Python SDK。")
    timeout_seconds = (
        CUSTOM_GATEWAY_REQUEST_TIMEOUT_SECONDS
        if is_custom_gateway(base_url)
        else OPENAI_REQUEST_TIMEOUT_SECONDS
    )
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout_seconds,
        max_retries=0,
    )
    if is_custom_gateway(base_url):
        candidate_models: list[str] = []
        for candidate in [model, *fallback_models]:
            if candidate and candidate not in candidate_models:
                candidate_models.append(candidate)
        fallback_messages = [{"role": "system", "content": system_prompt}]
        for item in history[-4:]:
            fallback_messages.append(
                {"role": item["role"], "content": trim_text(item["content"], 1200)}
            )
        fallback_messages.append({"role": "user", "content": compiled_user_prompt})
        errors: list[str] = []
        for candidate in candidate_models:
            try:
                completion = client.chat.completions.create(
                    model=candidate,
                    messages=fallback_messages,
                    temperature=0,
                    max_tokens=220,
                )
                content = completion.choices[0].message.content or ""
                return content.strip(), candidate
            except Exception as exc:
                errors.append(f"{candidate}: {exc.__class__.__name__}: {exc}")
        raise RuntimeError("；".join(errors))

    messages = build_api_messages(history, system_prompt, compiled_user_prompt)
    try:
        response = client.responses.create(model=model, input=messages)
        output_text = getattr(response, "output_text", "") or ""
        if output_text.strip():
            return output_text.strip(), model
    except Exception:
        pass

    try:
        fallback_messages = [{"role": "system", "content": system_prompt}]
        for item in history[-8:]:
            fallback_messages.append(
                {"role": item["role"], "content": item["content"]}
            )
        fallback_messages.append({"role": "user", "content": compiled_user_prompt})
        completion = client.chat.completions.create(
            model=model,
            messages=fallback_messages,
        )
        content = completion.choices[0].message.content or ""
        return content.strip(), model
    except Exception as exc:
        raise RuntimeError(f"模型调用失败：{exc}") from exc


def call_model_with_progress(
    status_placeholder: Any,
    api_key: str,
    base_url: str | None,
    model: str,
    fallback_models: list[str],
    system_prompt: str,
    history: list[dict[str, str]],
    compiled_user_prompt: str,
) -> tuple[str, str]:
    start_time = time.time()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(
        call_model,
        api_key=api_key,
        base_url=base_url,
        model=model,
        fallback_models=fallback_models,
        system_prompt=system_prompt,
        history=history,
        compiled_user_prompt=compiled_user_prompt,
    )
    try:
        while True:
            try:
                result = future.result(timeout=0.5)
                status_placeholder.empty()
                return result
            except concurrent.futures.TimeoutError:
                elapsed = int(time.time() - start_time)
                status_placeholder.info(
                    f"请求已发送，正在等待模型返回，已等待 {elapsed} 秒。"
                )
                if elapsed >= MODEL_TIMEOUT_SECONDS:
                    future.cancel()
                    status_placeholder.empty()
                    raise RuntimeError(
                        f"模型等待超过 {MODEL_TIMEOUT_SECONDS} 秒，已停止本次请求。"
                    )
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def render_header() -> None:
    st.title(APP_NAME)
    st.caption(APP_SUBTITLE)
    st.write(
        "围绕冻土概念、文献、时序数据、航测/DEM 诊断与学术写作，提供一个可直接扩展的研究型 agent 原型。"
    )


def render_chat_page(
    provider_label: str,
    model_name: str,
    api_key: str | None,
    base_url: str | None,
    fallback_models: list[str],
) -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.subheader("主聊天")
    task_mode = st.selectbox(
        "任务模式",
        options=list(TASK_MODE_CONFIG),
        format_func=lambda key: TASK_MODE_CONFIG[key]["label"],
    )
    research_context = st.text_area(
        "研究背景 / 区域 / 数据来源（可选）",
        height=120,
        placeholder="例如：青藏高原某湖岸，已有 InSAR 时序、两期 UAV 航测、局地气象与地温资料。",
    )
    uploads = st.file_uploader(
        "上传辅助材料（支持 txt / md / csv / xlsx / pdf / docx）",
        type=["txt", "md", "csv", "xlsx", "xls", "pdf", "docx"],
        accept_multiple_files=True,
    )

    extracted_files = [extract_uploaded_file(item) for item in uploads or []]
    if extracted_files:
        with st.expander("已解析的上传材料", expanded=False):
            for item in extracted_files:
                st.markdown(f"**{item.name}** ({item.kind})")
                st.code(item.preview)

    controls = st.columns([1, 1, 2])
    with controls[0]:
        if st.button("清空会话", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    with controls[1]:
        if st.button("重载知识库", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()
    with controls[2]:
        st.caption(
            f"Provider：`{provider_label}` | 模型：`{model_name}` | Base URL：`{base_url or '默认 OpenAI'}` | "
            f"API Key：{'已检测到' if api_key else '未配置，将使用离线提示'}"
        )

    for item in st.session_state.chat_history:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])

    prompt = st.chat_input("例如：两期航测 DoD 出现 0.2 m 湖岸后退，我该先排查哪些误差？")
    if not prompt:
        return

    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    query = f"{TASK_MODE_CONFIG[task_mode]['label']} {research_context} {prompt}"
    knowledge_hits = retrieve_knowledge(query)
    if is_custom_gateway(base_url):
        system_prompt = build_fast_system_prompt(task_mode)
        compiled_prompt = build_fast_compiled_prompt(
            user_prompt=prompt,
            task_mode=task_mode,
            research_context=research_context,
            files=extracted_files,
            knowledge_hits=knowledge_hits,
        )
    else:
        system_prompt = build_system_prompt(task_mode)
        compiled_prompt = build_compiled_prompt(
            user_prompt=prompt,
            task_mode=task_mode,
            research_context=research_context,
            files=extracted_files,
            knowledge_hits=knowledge_hits,
        )

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        if api_key and OpenAI is not None:
            try:
                reply, used_model = call_model_with_progress(
                    status_placeholder=status_placeholder,
                    api_key=api_key,
                    base_url=base_url,
                    model=model_name,
                    fallback_models=fallback_models,
                    system_prompt=system_prompt,
                    history=st.session_state.chat_history[:-1],
                    compiled_user_prompt=compiled_prompt,
                )
            except Exception as exc:
                used_model = None
                reply = (
                    f"模型调用失败：{exc}\n\n"
                    "当前已关闭自动重试；若仍失败，建议直接改用更快的候选模型。"
                )
        else:
            used_model = None
            reply = build_offline_reply(
                user_prompt=prompt,
                task_mode=task_mode,
                files=extracted_files,
                knowledge_hits=knowledge_hits,
            )
        st.markdown(reply)
        if used_model and used_model != model_name:
            st.caption(f"本轮请求已自动回退到更快可用模型：`{used_model}`")
        with st.expander("本轮命中的本地知识片段", expanded=False):
            if knowledge_hits:
                for index, hit in enumerate(knowledge_hits, start=1):
                    st.markdown(f"**{index}. {hit['source']}#chunk-{hit['chunk_id']}**")
                    st.code(trim_text(hit["text"], 1000))
            else:
                st.write("未命中明显相关片段。")

    st.session_state.chat_history.append({"role": "assistant", "content": reply})


def render_data_page() -> None:
    st.subheader("数据解读")
    st.write("上传 `csv/xlsx` 后，先做本地快速体检，再决定是否送入主聊天页做深度解释。")
    uploaded = st.file_uploader(
        "上传表格文件",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=False,
        key="data_page_uploader",
    )
    notes = st.text_area(
        "补充说明（可选）",
        height=100,
        placeholder="例如：第 1 列为时间，第 2 列为地温，第 3 列为 InSAR LOS 形变。",
    )
    if not uploaded:
        return

    try:
        df = load_dataframe_from_bytes(uploaded.name, uploaded.getvalue())
    except Exception as exc:
        st.error(f"表格解析失败：{exc}")
        return
    st.dataframe(df.head(20), use_container_width=True)
    st.markdown("**本地规则分析**")
    st.code(analyze_dataframe(df))
    if notes.strip():
        st.markdown("**补充说明**")
        st.write(notes.strip())

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        chart_cols = st.multiselect(
            "选择用于折线图的数值字段",
            options=numeric_cols,
            default=numeric_cols[: min(3, len(numeric_cols))],
        )
        if chart_cols:
            st.line_chart(df[chart_cols], use_container_width=True)


def render_uav_dem_page() -> None:
    st.subheader("航测 / DEM 诊断")
    st.write("用显式规则先评估 DoD 可靠性，再决定是否进入主聊天页做机理解释。")

    left, right = st.columns(2)
    with left:
        rtk_first = st.selectbox("第一期 RTK 条件", ["有", "无", "未知"])
        rtk_second = st.selectbox("第二期 RTK 条件", ["有", "无", "未知"])
        same_vertical_datum = st.selectbox("两期垂向基准一致吗", ["是", "否", "未知"])
        gcp_quality = st.selectbox("GCP / 检核点质量", ["高", "中", "低"])
        strip_shape = st.selectbox(
            "航带形态",
            ["块状覆盖", "规则长条航带", "不规则长条航带"],
        )
    with right:
        alignment_method = st.selectbox("配准方式", ["GCP+稳定区", "ICP/联合配准", "人工粗配准", "未知"])
        vegetation = st.selectbox("植被干扰", ["低", "中", "高"])
        water_edge = st.checkbox("是否包含水边/潮湿岸线")
        shadow_occlusion = st.checkbox("是否存在阴影或明显遮挡")
        stable_zone_checked = st.checkbox("是否已用稳定区做误差检核")

    max_change = st.number_input("观测到的最大绝对高程变化（m）", min_value=0.0, value=0.20, step=0.01)
    notes = st.text_area(
        "观测备注",
        height=120,
        placeholder="例如：2023 年无 RTK，2025 年有 RTK；长条航带沿湖岸展开，边缘变化明显。",
    )

    result = diagnose_uav_dem(
        {
            "rtk_first": rtk_first,
            "rtk_second": rtk_second,
            "same_vertical_datum": same_vertical_datum,
            "gcp_quality": gcp_quality,
            "strip_shape": strip_shape,
            "alignment_method": alignment_method,
            "vegetation": vegetation,
            "water_edge": water_edge,
            "shadow_occlusion": shadow_occlusion,
            "stable_zone_checked": stable_zone_checked,
            "max_change": max_change,
        }
    )

    st.markdown(f"**DoD 风险等级：{result['level']}**")
    st.markdown("**主要发现**")
    for item in result["findings"]:
        st.write(f"- {item}")
    st.markdown("**不确定性**")
    for item in result["uncertainties"]:
        st.write(f"- {item}")
    st.markdown("**建议动作**")
    for item in result["suggestions"]:
        st.write(f"- {item}")
    if notes.strip():
        st.markdown("**观测备注**")
        st.write(notes.strip())


def render_rules_page() -> None:
    st.subheader("规则库管理")
    rule_files = list_rule_files()
    if not rule_files:
        st.warning("当前没有找到 `rules/*.md` 文件。")
        return
    selected = st.selectbox(
        "选择规则文件",
        options=rule_files,
        format_func=lambda path: path.name,
    )
    editor_key = f"editor_{selected.name}"
    content = safe_read_text(selected)
    edited = st.text_area("规则内容", value=content, height=500, key=editor_key)

    columns = st.columns([1, 1, 2])
    with columns[0]:
        if st.button("保存规则", use_container_width=True):
            selected.write_text(edited, encoding="utf-8")
            st.success(f"已保存 {selected.name}")
    with columns[1]:
        if st.button("重载知识库索引", use_container_width=True):
            st.cache_resource.clear()
            st.success("知识库索引已清空，下次检索时会自动重建。")
    with columns[2]:
        st.caption("规则修改会直接写回当前项目目录。")

    st.markdown("**当前知识库文件**")
    knowledge_files = [
        str(path.relative_to(ROOT_DIR))
        for path in sorted(KNOWLEDGE_DIR.rglob("*"))
        if path.is_file()
    ]
    if knowledge_files:
        st.code("\n".join(knowledge_files))
    else:
        st.write("`knowledge/` 目录目前为空。")


def render_deploy_page() -> None:
    st.subheader("部署说明")
    st.markdown(
        """
**推荐目录结构**

```text
CryoScope-QTP/
├─ cryoscope_core.py
├─ cryoscope_qtp_streamlit_prototype_v2.py
├─ docs/
├─ tests/
├─ rules/
├─ knowledge/
├─ requirements.txt
└─ .streamlit/
   └─ secrets.toml
```

**安装依赖**

```bash
pip install -r requirements.txt
```

**启动**

```bash
streamlit run cryoscope_qtp_streamlit_prototype_v2.py
```

**配置 API Key（OpenAI 或 Kimi）**

优先使用环境变量：

```bash
export OPENAI_API_KEY="你的_key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="gpt-4.1-mini"
```

如果你要直接接 Kimi，也可以使用：

```bash
export KIMI_API_KEY="你的_kimi_key"
export KIMI_BASE_URL="https://api.moonshot.cn/v1"
export KIMI_MODEL="moonshot-v1-8k"
```

也可以在 `.streamlit/secrets.toml` 中配置：

```toml
OPENAI_API_KEY = "你的_key"
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_MODEL = "gpt-4.1-mini"

# 或者：
KIMI_API_KEY = "你的_kimi_key"
KIMI_BASE_URL = "https://api.moonshot.cn/v1"
KIMI_MODEL = "moonshot-v1-8k"
```
        """
    )


def main() -> None:
    st.set_page_config(page_title=APP_NAME, page_icon="🧊", layout="wide")
    render_header()

    settings = resolve_llm_settings(secret_reader=get_secret)
    api_key = settings.api_key
    base_url = settings.base_url
    default_model = settings.model
    fallback_models = settings.fallback_models

    with st.sidebar:
        st.header("导航")
        page = st.radio(
            "页面",
            ["主聊天", "数据解读", "航测/DEM诊断", "规则库", "部署说明"],
        )
        model_name = st.text_input("模型名称", value=default_model)
        configured_base_url = st.text_input("Base URL", value=base_url or "")
        runtime_provider = infer_llm_provider(
            base_url=configured_base_url or None,
            model=model_name,
        )
        if not fallback_models:
            fallback_models = get_default_fallback_models(
                runtime_provider,
                configured_base_url or None,
            )
        st.caption("Fallback: " + (", ".join(fallback_models) if fallback_models else "无"))
        st.caption(f"知识库目录：`{KNOWLEDGE_DIR}`")
        if api_key:
            st.success("已检测到 API Key")
        else:
            st.info("未检测到 API Key，主聊天页将使用离线提示模式。")

    if page == "主聊天":
        render_chat_page(
            provider_label="Kimi" if runtime_provider == "kimi" else "OpenAI-compatible",
            model_name=model_name,
            api_key=api_key,
            base_url=configured_base_url or None,
            fallback_models=fallback_models,
        )
    elif page == "数据解读":
        render_data_page()
    elif page == "航测/DEM诊断":
        render_uav_dem_page()
    elif page == "规则库":
        render_rules_page()
    else:
        render_deploy_page()


if __name__ == "__main__":
    main()

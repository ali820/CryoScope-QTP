# CryoScope-QTP

一个基于共享对话落地的冻土研究 agent 原型，面向青藏高原多年冻土、湖岸演化、InSAR、UAV/DEM 差分和学术写作场景。

仓库附带了一组可直接运行的示例规则与知识片段，方便快速启动，并可按自己的研究主题继续扩展。

## 当前能力

- 主聊天页：支持任务模式切换、上传文件注入、本地轻量 RAG、OpenAI API 调用
- 数据解读页：对 `csv/xlsx` 做本地快速体检
- 航测 / DEM 诊断页：基于显式规则判断两期 DoD 风险
- 规则库页：直接查看和修改 `rules/*.md`

## 目录结构

```text
.
├─ cryoscope_core.py
├─ cryoscope_qtp_streamlit_prototype_v2.py
├─ docs/
├─ scripts/
├─ tests/
├─ rules/
├─ knowledge/
├─ requirements.txt
└─ .streamlit/
   └─ secrets.toml.example
```

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 启动

```bash
streamlit run cryoscope_qtp_streamlit_prototype_v2.py
```

或：

```bash
bash scripts/run_local.sh
```

## API 配置

优先使用环境变量：

```bash
export OPENAI_API_KEY="你的_key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="gpt-4.1-mini"
export OPENAI_FALLBACK_MODELS="gpt-4.1-mini"
```

如果你要直接接 Kimi，也可以使用：

```bash
export KIMI_API_KEY="你的_kimi_key"
export KIMI_BASE_URL="https://api.moonshot.cn/v1"
export KIMI_MODEL="moonshot-v1-8k"
export KIMI_FALLBACK_MODELS="moonshot-v1-8k"
```

同时兼容 `MOONSHOT_API_KEY` / `MOONSHOT_BASE_URL` / `MOONSHOT_MODEL` / `MOONSHOT_FALLBACK_MODELS` 这一组别名。

也可以复制 `.streamlit/secrets.toml.example` 为 `.streamlit/secrets.toml` 后填写。

## 项目文档

- 版本范围：`docs/v1_scope.md`
- 上线检查清单：`docs/release_checklist.md`

## 最小测试

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

## 本地校验

```bash
bash scripts/verify_local.sh
```

## 使用建议

1. 先把你自己的论文笔记、规范摘录、站点说明放进 `knowledge/`
2. 再把你的判断规则补进 `rules/`
3. 然后在主聊天页上传实际文件，让 agent 基于本地知识和上传材料联合回答

## 注意

- 当前 RAG 是轻量关键词召回，不是向量数据库。
- `PDF/DOCX` 提取更适合普通文本型文档；扫描版 PDF 仍可能失败。
- 航测 / DEM 页目前是规则驱动，适合做第一轮质量筛查，不替代正式误差分析。
- 建议通过环境变量或本地 `.streamlit/secrets.toml` 配置私有密钥，不要提交包含真实凭据的文件。

## License

This project is released under the MIT License.

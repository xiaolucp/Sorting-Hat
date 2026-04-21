# Sorting Hat — 面试问题意图识别

面试辅助场景下，对面试官提出的问题做意图分类，用于下游路由到不同的提示词策略、检索链路和答案生成方式。

---

## 现状（2026-04-21）

### 已完成

**数据收集**
- BigQuery 拉取 2026-04-01 ~ 2026-04-20 共 20 天的 FRAI 会话数据
- 原始文件：`data/raw/frai_sessions_YYYY-MM-DD.json`（每天约 200-500 sessions）
- 扁平化 turn 文件：`data/raw/frai_turns_YYYY-MM-DD.jsonl`（含 trace_id、role、timestamps 等）

**LLM 预标注 Pipeline**（`src/sorting_hat/labeling/`）
- `schema.py`：Pydantic 模型 — `TurnInput`、`SessionContext`、`LabelResult`、`LabeledTurn`
- `prompts.py`：系统提示词（5 类决策树）+ 用户模板，独立文件以稳定 prompt cache
- `labeler.py`：`IntentLabeler` 类，通过 LiteLLM 代理调用 `azure/gpt-4o`
  - 强制 JSON 输出（`response_format={"type": "json_object"}` + Pydantic 验证）
  - 全程使用 streaming 累积（代理非流式路径会返回空 content）
  - 3 次重试 + 指数退避（2s、4s）
  - `_merge_consecutive()`：合并同 role 在 30s 内的相邻 turn，压缩上下文碎片
  - `_truncate_middle()`：超长 turn 保留首尾，截断中间（保留话题开头 + 约束条件结尾）
  - `_format_prior_turns()`：取 `max_turns` 个已合并的历史 slot（默认 6 个）

**标注脚本**（`scripts/label_sessions.py`）
- 读取 `data/raw/frai_sessions_YYYY-MM-DD.json` → 写入 `data/labeled/turns_YYYY-MM-DD.jsonl`
- 支持断点续跑（按 turn_id 去重跳过）
- 参数：`--date`、`--model`、`--modes`（过滤 interview_mode）、`--sample N`（随机采样）、`--workers`（并发，默认 8）、`--limit`、`--min-len`、`--force`
- 上下文拉取策略：每个 turn 取前 `context_window * 4` 条原始 turn → 合并 → 展示最后 6 个合并后 slot
- 已标注：`data/labeled/turns_2026-04-20.jsonl`（111 条，copilot 模式抽样）

**Streamlit 会话查看器**（`scripts/session_chat_viewer.py`）
- 从 `data/raw/frai_sessions_*.json` 读取所有会话，侧边栏筛选日期/模式/语言
- 自动加载对应日期的 `data/labeled/turns_*.jsonl`，在聊天气泡上叠加颜色标签
- 标签颜色：coding=绿（`#c6f6d5`）/ system_design=蓝（`#bfdbfe`）/ project_qa=紫（`#e9d5ff`）/ chat=黄（`#fef08a`）/ no_answer_needed=灰（`#e2e8f0`）
- 侧边栏：`Has labeled turns` 过滤、多选意图标签过滤、图例
- 运行：`uv run streamlit run scripts/session_chat_viewer.py`

**训练脚本**（`scripts/train_qwen_lora.py`）
- 基础模型：`Qwen/Qwen2.5-0.5B-Instruct`（默认）
- LoRA（r=16, alpha=32）+ `trl.SFTTrainer`，只对 assistant response 计算 loss
- 种子数据：`data/training/seed_v1.jsonl`（20 条手工样本，格式见下）
- 已有一次 smoke-test 产物：`models/qwen-intent-lora-v1/`（Qwen2.5-0.5B + LoRA，仅供验证流程可跑通，不具实际分类能力）
- 运行：`uv run --extra train python scripts/train_qwen_lora.py --data data/training/seed_v1.jsonl --model Qwen/Qwen2.5-0.5B-Instruct --output models/qwen-intent-lora-v1`

---

### 标签定义（当前版本）

| label | 含义 |
|---|---|
| `coding` | 写代码 / 描述算法 / trace 执行；预期答案是可运行代码 |
| `system_design` | 设计多组件端到端系统（含规模、trade-off）；"解释 X 是什么"不算 |
| `project_qa` | 追问候选人过去的项目 / 工作经历 / STAR 式行为面问题 |
| `chat` | 概念解释、小知识 Q&A、闲聊、澄清；不属于上面三类的一切问题 |
| `no_answer_needed` | 不需要候选人回答：口水话、ASR 噪音、面试官自言自语、麦克风检查 |

辅助字段：`secondary_label`（混合题次要类别，nullable）、`confidence`（0-1）、`reason`（一句话）

---

### 训练数据格式（`data/training/seed_v1.jsonl`）

每行一条 JSON：
```json
{
  "prior_turns": [{"role": "interviewer", "text": "..."}],
  "current_turn": {"role": "interviewer", "text": "...", "source": "...", "trace_id": "..."},
  "session": {"interview_mode": "code", "programming_language": "Python", "goal_position": "...", "goal_company": "..."},
  "label": "coding",
  "confidence": 0.95,
  "reason": "...",
  "secondary_label": null
}
```

---

### 关键路径

```
data/raw/frai_sessions_YYYY-MM-DD.json   ← BigQuery (scripts/pull_sessions.py)
        ↓
scripts/label_sessions.py               ← LiteLLM 代理 (azure/gpt-4o)
        ↓
data/labeled/turns_YYYY-MM-DD.jsonl     ← 标注结果
        ↓
scripts/session_chat_viewer.py          ← 人工审核 / 抽检 (Streamlit)
        ↓
data/training/*.jsonl                   ← 精选训练集
        ↓
scripts/train_qwen_lora.py              ← Qwen LoRA SFT (需 GPU)
        ↓
models/qwen-intent-lora-v1/             ← LoRA adapter
```

---

### 环境与工具链

- 包管理：`uv`（`pyproject.toml` + `uv.lock`）
- Python：3.11
- 基础依赖：`uv sync`（litellm、pydantic、tqdm、pandas 等）
- 训练依赖：`uv sync --extra train`（torch、transformers、peft、trl、accelerate、datasets）
- 环境变量（`.env.local`，不入 git）：
  - `LITELLM_BASE_URL`：LiteLLM 代理地址（`https://dev-litellm.frai.pro`）
  - `LITELLM_API_KEY`：代理鉴权 key
  - `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL`（可选，查 trace 用）
  - `GOOGLE_APPLICATION_CREDENTIALS`：BigQuery 服务账号 JSON 路径

---

### A10 开发机迁移清单

1. **推代码**：`git push` 到远程，A10 上 `git clone`
2. **传数据**：`rsync -av data/raw/ a10:/path/to/Sorting-Hat/data/raw/`（data 目录不入 git）
3. **传已标注数据**：`rsync -av data/labeled/ a10:/path/to/Sorting-Hat/data/labeled/`
4. **传种子数据**：`rsync -av data/training/ a10:/path/to/Sorting-Hat/data/training/`
5. **配置 .env.local**：在 A10 上手动写入 API keys
6. **安装依赖**：`uv sync --extra train`（会自动下载 CUDA 版 PyTorch）
7. **验证训练流程**：`uv run --extra train python scripts/train_qwen_lora.py --data data/training/seed_v1.jsonl --no-smoke-test`
8. **扩规模标注**：在本机跑 `label_sessions.py` 继续积累 `data/labeled/`，再 rsync 到 A10 做训练

---

## 为什么存在

当前系统直接用大模型做意图识别，存在两个问题：
1. **准确率不稳定**：口语化表达、混合背景、追问、补充限制条件等多样输入，大模型分类不够稳
2. **成本与延迟**：作为高频第一层判断模块，直接调用 LLM 不合适

本项目训练一个**小型生成式分类模型**承担这一层，必要时将低置信样本 fallback 给大模型。

---

## 技术方案

### 模型

- **基础模型**：Qwen（尺寸待 benchmark，候选 0.5B / 1.5B / 3B）
- **微调方式**：LoRA
- **输出格式**：结构化 JSON `{label, secondary_label, confidence, reason}`

### 选型理由

选 Qwen + LoRA 而非 BERT / BGE + 分类头：
- 生成式一次推理可输出 **标签 + 理由 + 置信度 + 混合标签**，下游路由和 badcase 分析更友好
- 多轮对话和复杂上下文下，生成式通常优于 encoder-only
- BERT 路线要自己加置信度头、多标签头，工程复杂度反而更高

**注意**：Qwen 对多轮/混合的"天然支持"是基础能力层面的，**具体能力要靠训练数据教**——训练集必须显式包含多轮格式、带 secondary_label 的混合样本、话题切换样本。

---

## 数据 Pipeline

### 流程

1. **原始数据收集** → `data/raw/` （`scripts/pull_sessions.py`）
2. **LLM 预标注**：`scripts/label_sessions.py` 调用 LiteLLM 代理批量标注
   - 先按 `--modes copilot code` 过滤高质量模式，跳过 phone 等噪声较多的模式
   - `--sample N` 随机采样控制规模，`--workers 20` 并发
3. **人工审核**：`scripts/session_chat_viewer.py` 查看彩色标签，标记 badcase
4. **精选进训练集**：高置信 + 人工抽检通过 → `data/training/`
5. **Golden Set**：500–1000 条全人工标注，**永不进训练**，作为最终 benchmark（待建）

### 数据切分原则

- 按类别分层（stratified）
- **按 session 切**：多轮对话整个 session 同进同出，避免泄漏
- 按来源分层：标准题、真实面试转录、合成数据按比例分布到 train / val / test

---

## 评估（待定）

- 按类别 precision / recall + 混淆矩阵（重点看 `coding ↔ system_design`）
- 置信度校准（ECE），决定 LLM fallback 阈值
- 分层报告：干净样本 / 口语样本 / 多轮追问 / 对抗样本
- 线上指标：下游回答满意度、用户重问率、P95 延迟

---

## 待办

- [ ] 扩大标注规模：对全部 20 天数据跑 `label_sessions.py`（copilot + code 模式，每天 ~500 turns）
- [ ] 人工审核已标注数据，修正 badcase，建立 golden set
- [ ] 在 A10 上跑 Qwen2.5-1.5B / 3B 的训练和 benchmark
- [ ] 标签边界规则文档（LRU=coding、rate limiter=system_design、"如何设计但要求写代码"=mixed）
- [ ] 评估脚本（混淆矩阵、ECE、分层报告）
- [ ] 在线接入架构（分层 fallback：小模型低置信 → LLM）

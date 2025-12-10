# ☕ All-in RAG for Halter - Coffee Machine Assistant

企业级 RAG (Retrieval-Augmented Generation) 系统，带有智能路由和 Streamlit 前端界面。

## 🌟 功能特性

- **智能路由**：基于 LLM 的查询路由，自动决定走 RAG、升级工单、查询数据库或调用 API
- **混合检索**：结合稠密检索（Dense）+ 稀疏检索（BM25）+ RRF 融合 + BGE 重排
- **多模态支持**：文本和图像的分离 embedding 和检索
- **置信度评估**：三层置信度检测（检索层、回答层、评审层）
- **LangChain 工具**：将 RAG 和工单升级包装为 LangChain Tools
- **Streamlit 前端**：友好的 Web 界面

## 📁 项目结构

```
all-in-rag-for-halter/
├── agent/                      # Agent 主入口
│   └── main.py                # 命令行 agent 入口
├── tools/                      # LangChain 工具
│   ├── __init__.py
│   ├── rag_tool.py            # RAG 检索工具
│   └── escalate_tool.py       # 工单升级工具
├── utils/                      # 核心工具模块
│   ├── chunker.py             # Markdown 文档切块
│   ├── embed_text_chroma.py   # 文本 embedding
│   ├── embed_image_chroma.py  # 图像 embedding
│   ├── hybrid_retrieve.py     # 混合检索 + LLM 回答
│   ├── router_chain.py        # LLM 路由决策
│   ├── prompt_config.py       # Prompt 模板管理
│   └── llm_answer.py          # LLM 调用封装
├── tests/                      # 测试脚本
│   └── test_embedded_dim_chroma.py
├── data/                       # 数据目录
│   ├── markdown/              # 处理后的 Markdown 文件
│   └── images/                # 下载的图片 (gitignore)
├── chroma_db/                  # Chroma 向量数据库 (gitignore)
├── logs/                       # 日志和工单记录 (gitignore)
├── app.py                      # Streamlit 前端应用 ⭐
├── requirements.txt            # Python 依赖
├── .env                        # API 密钥配置 (gitignore)
└── .gitignore

```

## 🚀 快速开始

### 1. 安装依赖

```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env` 文件，添加你的 API 密钥：

```env
OPENAI_API_KEY=sk-your-openai-api-key-here
```

### 3. 准备数据

```bash
# 1. 使用 MinerU 处理文档（生成 Markdown）
# （假设你已经有了处理好的 Markdown 文件）

# 2. 切分文档
python utils/chunker.py

# 3. 生成文本 embedding
python utils/embed_text_chroma.py \
  --chunks data/chunks.jsonl \
  --collection coffee_text \
  --delete-source "MinerU_markdown_*.md"

# 4. 生成图像 embedding
python utils/embed_image_chroma.py \
  --markdown data/markdown/MinerU_markdown_*.md \
  --collection coffee_images \
  --delete-source "MinerU_markdown_*.md"
```

### 4. 启动 Streamlit 应用 🎉

```bash
streamlit run app.py
```

应用将在浏览器中自动打开（默认 `http://localhost:8501`）。

## 🎨 使用 Streamlit 界面

### 主界面
- **聊天输入框**：在底部输入你的问题
- **聊天历史**：查看完整的对话历史，包括置信度和来源引用
- **侧边栏设置**：调整 RAG 参数和路由行为

### 侧边栏功能

1. **RAG Configuration**
   - `Top Rerank Results`：重排后使用的文档数量（1-10）
   - `Neighbor Radius`：扩展的邻居块数量（0-3）
   - `Max Context Tokens`：LLM 上下文的最大 token 数（500-3000）

2. **Router Override**
   - 启用手动检索信号，用于测试和调试
   - 手动设置 `top1`、`avg_top5`、`hits` 来控制路由决策

3. **Clear Chat History**
   - 清空聊天记录

### 消息类型

- 🟦 **用户消息**：蓝色边框
- 🟢 **RAG 回答**：绿色边框，显示置信度和来源
- 🟠 **升级工单**：橙色边框，显示工单号
- 🔴 **错误消息**：红色边框

## 🔧 命令行使用

如果你更喜欢命令行，可以直接使用：

```bash
# 完整的 Agent 流程
python agent/main.py \
  --query "my coffee is not hot" \
  --top1 0.9961 \
  --avg-top5 0.646 \
  --hits 6 \
  --sections TROUBLESHOOTING

# 仅测试路由决策
python utils/router_chain.py \
  --query "how to clean the machine" \
  --top1 0.8 \
  --avg-top5 0.6 \
  --hits 5

# 仅测试 RAG 检索
python utils/hybrid_retrieve.py \
  --query "coffee is weak" \
  --top-rerank 3 \
  --neighbor-radius 1 \
  --max-context-tokens 1500
```

## 🧠 系统架构

### 路由逻辑

1. **Router** (`utils/router_chain.py`)
   - 接收用户查询和检索信号（可选）
   - 硬编码规则优先：
     - 高分 → RAG
     - 低分 → 升级
   - LLM 意图判断（如果规则不匹配）
   - 输出：`{action, confidence, reason}`

2. **Action 类型**
   - `rag`：使用 RAG Tool 检索和回答
   - `escalate`：使用 Escalate Tool 创建工单
   - `db`：查询数据库（待实现）
   - `api`：调用外部 API（待实现）

### RAG Pipeline

1. **稠密检索**：使用 `text-embedding-3-large` 从 Chroma 检索
2. **稀疏检索**：使用 BM25 关键词匹配
3. **RRF 融合**：合并两种检索结果
4. **BGE 重排**：使用 `BAAI/bge-reranker-base` 重新排序
5. **邻居扩展**：收集上下文邻居块，限制 token 数
6. **LLM 回答**：使用 `gpt-4o-mini` 生成答案（JSON 格式）
7. **Judge 评审**：第二次 LLM 调用，评估答案质量

### 置信度评估

- **检索层**：`top1 < 0.35` or `avg_top5 < 0.30` → 低置信度
- **回答层**：LLM 输出 `can_answer: false` or `confidence < 0.5`
- **评审层**：Judge 输出 `is_supported: false` or `hallucination_level >= 1`

## 📊 工单日志

所有升级的工单会记录在 `logs/tickets.jsonl` 中，格式：

```json
{
  "ticket_id": "TICKET-20251210-A1B2C3D4",
  "query": "用户的问题",
  "reason": "升级原因",
  "timestamp": "2025-12-10T14:53:00.123456",
  "status": "open",
  "assigned_to": "help_desk"
}
```

## 🐛 故障排查

### 1. 控制台编码错误（Windows GBK）

如果遇到 `UnicodeEncodeError`，`safe_print` 函数会自动处理，将特殊字符替换为 `?`。

### 2. Chroma 连接问题

确保 `chroma_db/` 目录存在，且有读写权限。

### 3. OpenAI API 超时

检查网络连接，或在 `.env` 中配置代理（如果需要）。

### 4. 模型下载慢

- 文本 embedding：使用 OpenAI API，无需下载
- 图像 embedding：OpenCLIP 模型会在首次使用时下载，建议使用 `ViT-B-32`（较小）
- BGE Reranker：首次使用时会从 HuggingFace 下载

## 📝 开发说明

### 添加新工具

1. 在 `tools/` 目录创建新的工具文件
2. 继承 `langchain.tools.BaseTool`
3. 实现 `_run()` 方法
4. 在 `app.py` 中集成新工具

### 修改 Prompt

所有 Prompt 模板在 `utils/prompt_config.py` 中统一管理：
- `build_messages_answer`：RAG 回答 prompt
- `build_messages_judge`：Judge 评审 prompt
- `build_router_messages`：Router 路由 prompt

### 调整路由规则

在 `utils/router_chain.py` 的 `main()` 函数中修改硬编码规则：

```python
if top1 >= 0.7 and avg5 >= 0.5 and hits >= 3:
    result = {"action": "rag", "confidence": 0.9, "reason": "High retrieval scores"}
    # ...
```

## 📄 License

MIT

## 🙏 致谢

- [LangChain](https://www.langchain.com/)
- [OpenAI](https://openai.com/)
- [Chroma](https://www.trychroma.com/)
- [Streamlit](https://streamlit.io/)
- [MinerU](https://github.com/opendatalab/MinerU)

---

**Enjoy your RAG system! ☕✨**

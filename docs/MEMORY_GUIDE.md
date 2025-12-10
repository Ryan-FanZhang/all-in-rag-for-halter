# 💭 Memory Integration Guide

本指南介绍如何使用 LangChain Memory 实现多轮对话和上下文理解。

## 🧠 Memory 架构

### 核心组件

```
┌─────────────────────────────────────────┐
│          Streamlit Session              │
│  ┌─────────────────────────────────┐   │
│  │    Memory Manager               │   │
│  │  - ConversationBufferWindow (5) │   │
│  │  - Full History Storage         │   │
│  │  - Context Provider             │   │
│  └─────────────────────────────────┘   │
│               ↕                         │
│  ┌──────────┬──────────┬──────────┐   │
│  │   RAG    │   API    │ Escalate │   │
│  │  Tool    │   Tool   │   Tool   │   │
│  └──────────┴──────────┴──────────┘   │
└─────────────────────────────────────────┘
```

### Memory Manager 特性

1. **ConversationBufferWindowMemory**
   - 保留最近 5 轮对话（10 条消息）
   - 自动管理窗口大小
   - 提供上下文给 LLM

2. **Full History Storage**
   - 完整保存所有对话
   - 包含元数据（timestamp, action, confidence）
   - 支持序列化和恢复

3. **Context Provider**
   - 为所有工具提供对话历史
   - 格式化为 LLM 可读的消息列表
   - 支持自定义系统提示

---

## 🚀 功能演示

### 1. 多轮对话（Multi-turn Conversation）

#### 场景：订单查询 + 追问

**Turn 1**:
```
User: what is the status of my order ORD12345
Assistant: Your order ORD12345 has been delivered! 🎉
(Action: api, Confidence: 0.85)
```

**Turn 2** (追问):
```
User: when was it delivered?
Assistant: Based on the order information, your order was delivered 
on December 3, 2025. It was shipped from our warehouse and tracked 
with number TRK946458903.
(Action: api, 利用 memory 知道 "it" 指的是 ORD12345)
```

**Turn 3** (再追问):
```
User: can I return it?
Assistant: Let me check the return policy for your Coffee Machine ECAM23.420...
(Action: rag, 知道要查询的是这个产品的退货政策)
```

---

### 2. 上下文延续（Context Continuation）

#### 场景：故障排查 + 详细询问

**Turn 1**:
```
User: my coffee is not hot
Assistant: To get hotter coffee, you can:
1. Preheat the cups
2. Run a rinse cycle
3. Adjust temperature settings
(Action: rag)
```

**Turn 2** (延续上下文):
```
User: how do I adjust the temperature?
Assistant: To adjust the coffee temperature:
1. Turn off the machine (standby)
2. Press the main switch
3. Hold the temperature icon
4. Select: low/medium/high/maximum
(Action: rag, 知道是在讨论咖啡温度调节)
```

**Turn 3** (继续深入):
```
User: what's the maximum temperature?
Assistant: The manual doesn't specify the exact maximum temperature 
in degrees, but it's the highest setting available. It will heat 
the coffee to the hottest possible level safe for consumption.
(Action: rag, 结合前面的温度讨论)
```

---

### 3. 指代消解（Coreference Resolution）

#### 场景：产品咨询 + 代词使用

**Turn 1**:
```
User: tell me about the Coffee Machine ECAM23.420
Assistant: The De'Longhi ECAM23.420 is a bean to cup espresso machine...
Price: $599.99, Power: 1450W, Pressure: 15 bar
(Action: api/rag)
```

**Turn 2** (使用 "it"):
```
User: is it in stock?
Assistant: Yes! The Coffee Machine ECAM23.420 is currently in stock.
Available quantity: 45 units across 2 warehouses.
(Action: api, "it" → ECAM23.420)
```

**Turn 3** (使用 "this machine"):
```
User: how do I clean this machine?
Assistant: To clean the Coffee Machine ECAM23.420:
1. Coffee grounds container...
2. Drip tray...
(Action: rag, "this machine" → ECAM23.420)
```

---

### 4. 跨工具记忆（Cross-tool Memory）

#### 场景：API + RAG + Escalate

**Turn 1** (API):
```
User: check my order ORD12345
Assistant: Order delivered, tracking: TRK946458903
(Action: api)
```

**Turn 2** (RAG):
```
User: the machine from that order is broken
Assistant: Let me help you troubleshoot. What's the issue?
(Action: rag, 知道是 Coffee Machine ECAM23.420)
```

**Turn 3** (Escalate):
```
User: I want a refund
Assistant: I've created ticket TICKET-20251210-ABC123 for your 
refund request regarding order ORD12345.
(Action: escalate, 记住订单号和产品)
```

---

## 🎯 Memory 使用示例

### 在 Streamlit 中查看 Memory

侧边栏会显示 **💭 Memory Summary**（可展开）：

```
Total conversation turns: 3

Recent exchanges:
👤 User: what is my order status
🤖 Assistant: Your order ORD12345 has been delivered!
👤 User: when was it delivered?
🤖 Assistant: It was delivered on December 3, 2025...
👤 User: can I return it?
🤖 Assistant: Let me check the return policy...
```

---

## 🛠️ 技术实现

### Memory Manager API

```python
from utils.memory_manager import create_memory_manager

# 创建 memory manager
memory = create_memory_manager(window_size=5)

# 添加用户消息
memory.add_user_message("what is my order status")

# 添加 AI 消息
memory.add_ai_message("Your order has been delivered", metadata={
    "action": "api",
    "confidence": 0.85
})

# 获取对话历史（用于 LLM）
messages = memory.get_context_for_llm(
    current_query="when was it delivered?",
    system_prompt="You are a helpful assistant..."
)

# 获取摘要
summary = memory.get_summary()

# 清空历史
memory.clear()
```

### 在工具中使用 Memory

#### 示例：API Tool with Memory

```python
# Before (no memory)
llm.invoke([
    SystemMessage("You are a helpful assistant"),
    HumanMessage(f"Summarize: {api_data}")
])

# After (with memory)
messages = memory.get_context_for_llm(
    current_query=f"Summarize: {api_data}",
    system_prompt="You are a helpful assistant"
)
llm.invoke(messages)
```

---

## 📊 Memory 配置

### Window Size（窗口大小）

```python
# 在 app.py 中配置
memory = create_memory_manager(window_size=5)
```

| Window Size | 保留对话轮数 | 适用场景 |
|------------|------------|---------|
| 3 | 6 条消息 | 短对话，快速响应 |
| 5 (默认) | 10 条消息 | 标准对话，平衡性能 |
| 10 | 20 条消息 | 长对话，深度上下文 |

### 持久化

Memory 自动保存在 `st.session_state` 中：
- ✅ 刷新页面不丢失
- ✅ 支持导出/导入
- ❌ 关闭浏览器会清空（可以后续添加数据库持久化）

---

## 🧪 测试 Memory 功能

### 测试用例 1：订单追问

```
1. "what is the status of my order ORD12345"
   → 期待：API 返回订单状态

2. "when will it arrive?"
   → 期待：知道 "it" = ORD12345，给出预计送达时间

3. "can I track it?"
   → 期待：给出 tracking number TRK946458903
```

### 测试用例 2：故障排查深入

```
1. "my coffee is weak"
   → 期待：RAG 返回调节建议

2. "how do I adjust the grinder?"
   → 期待：详细的研磨调节步骤

3. "what if that doesn't work?"
   → 期待：提供其他解决方案或升级
```

### 测试用例 3：跨工具记忆

```
1. "check inventory for Coffee Machine ECAM23.420"
   → 期待：API 返回库存

2. "how does this machine work?"
   → 期待：RAG 返回使用说明（知道是 ECAM23.420）

3. "I want to buy it"
   → 期待：API 创建订单或引导购买
```

---

## 🎉 Memory 带来的提升

| 功能 | 无 Memory | 有 Memory |
|------|----------|----------|
| **追问** | ❌ 不理解上下文 | ✅ 理解上下文 |
| **代词** | ❌ "it" 是什么？ | ✅ 自动识别指代 |
| **连贯性** | ❌ 每次都是新对话 | ✅ 连续对话 |
| **用户体验** | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **智能度** | 基础 | 高级 |

---

## 🔧 高级功能（未来扩展）

### 1. 对话总结（Conversation Summary）
```python
from langchain.memory import ConversationSummaryMemory

# 自动总结长对话，节省 token
summary_memory = ConversationSummaryMemory(llm=llm)
```

### 2. 实体提取（Entity Extraction）
```python
# 自动提取关键实体（订单号、产品名、时间）
entities = {
    "order_id": "ORD12345",
    "product": "Coffee Machine ECAM23.420",
    "tracking": "TRK946458903"
}
```

### 3. 数据库持久化
```python
# 保存到 SQLite/PostgreSQL
memory.save_to_db(session_id="user_123")
memory = load_from_db(session_id="user_123")
```

### 4. 多用户会话
```python
# 每个用户独立的 memory
memories = {}
memories[user_id] = create_memory_manager()
```

---

## 📝 最佳实践

1. **定期清理**：长对话后清空 memory，避免 token 超限
2. **敏感信息**：不要在 memory 中存储敏感数据（密码、信用卡）
3. **测试追问**：确保代词和指代正确解析
4. **性能监控**：观察 token 使用量，调整 window_size

---

**Enjoy contextual conversations! 🚀**


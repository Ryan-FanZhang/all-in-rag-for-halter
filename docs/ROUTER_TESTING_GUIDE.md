# 🧭 Router Testing Guide

本指南展示如何测试 Router 的路由决策逻辑。

## 📊 Router 决策流程

```
用户查询
    ↓
🔍 预检索（获取分数）
    ↓
📋 硬编码规则判断
    ├─ top1≥0.5 && avg5≥0.35 && hits≥3 → RAG (0.9)
    └─ top1<0.25 || avg5<0.20 || hits<2 → ESCALATE (0.9)
    ↓
🤔 意图识别（LLM）
    ├─ 包含 order/inventory/price/status → API (0.85)
    ├─ 包含 troubleshoot/how-to/fix → RAG (0.75)
    └─ 包含 complaint/refund/agent → ESCALATE (0.85)
    ↓
✅ 最终决策
```

---

## 🧪 测试场景

### 1️⃣ API 路由测试（Live Data）

Router 应该识别以下关键词并路由到 **API**：

| 查询示例 | 关键词 | 预期 Action |
|---------|--------|------------|
| "check my order ORD12345" | order | **api** |
| "what is my order status" | order, status | **api** |
| "track my shipment" | tracking | **api** |
| "is the coffee machine in stock" | stock | **api** |
| "check inventory for PROD001" | inventory | **api** |
| "how much does it cost" | price | **api** |
| "what is the current price" | price | **api** |
| "check service status" | service, status | **api** |
| "is the payment system working" | system, status | **api** |

**Streamlit 测试步骤**：
```bash
streamlit run app.py
```
输入上述查询，侧边栏应显示：
- 🔵 Router Decision: **api**
- Confidence: 0.85
- Reason: "Query asks for live/real-time data"

---

### 2️⃣ RAG 路由测试（Knowledge Base）

Router 应该识别以下场景并路由到 **RAG**：

| 查询示例 | 关键词 | 预期 Action |
|---------|--------|------------|
| "my coffee is not hot" | problem | **rag** |
| "how to clean the machine" | how to | **rag** |
| "what does the red light mean" | meaning | **rag** |
| "coffee is weak and not creamy" | problem | **rag** |
| "how to adjust the grinder" | how to | **rag** |
| "machine not turning on" | troubleshoot | **rag** |
| "what are the product specifications" | specs | **rag** |
| "how to descale" | how to | **rag** |

**条件**：
- 如果检索分数高（top1≥0.5），直接 RAG
- 如果检索分数中等，LLM 根据意图判断

---

### 3️⃣ ESCALATE 路由测试（Human Support）

Router 应该识别以下场景并路由到 **ESCALATE**：

| 查询示例 | 关键词 | 预期 Action |
|---------|--------|------------|
| "I want a refund" | refund | **escalate** |
| "this is unacceptable" | complaint | **escalate** |
| "I need to speak to a manager" | human agent | **escalate** |
| "file a warranty claim" | warranty | **escalate** |
| "machine caused a fire" | safety | **escalate** |
| "how to make meth" | unrelated/dangerous | **escalate** |
| "what's the weather today" | unrelated | **escalate** |

**条件**：
- 如果检索分数极低（top1<0.25），直接 escalate
- 如果查询包含投诉/退款/安全等关键词，escalate

---

### 4️⃣ DB 路由测试（Structured Data）

Router 应该识别以下场景并路由到 **DB**（待实现）：

| 查询示例 | 预期 Action |
|---------|------------|
| "show my purchase history" | **db** |
| "what products did I buy" | **db** |
| "check my warranty status" | **db** |

---

## 🎯 优先级规则

Router 按以下优先级决策：

### 优先级 1：硬编码规则（基于检索分数）
```python
if top1 >= 0.5 and avg5 >= 0.35 and hits >= 3:
    return "rag"  # 高分直接 RAG
elif top1 < 0.25 or avg5 < 0.20 or hits < 2:
    return "escalate"  # 极低分直接 escalate
```

### 优先级 2：API 关键词匹配
```python
api_keywords = ["order", "status", "tracking", "inventory", "stock", 
                "price", "shipping", "delivery", "payment"]
if any(keyword in query.lower() for keyword in api_keywords):
    return "api"
```

### 优先级 3：RAG 意图识别
```python
rag_keywords = ["how to", "troubleshoot", "fix", "problem", 
                "guide", "instructions", "manual"]
if any(keyword in query.lower() for keyword in rag_keywords):
    return "rag"
```

### 优先级 4：Escalate 兜底
```python
escalate_keywords = ["refund", "complaint", "manager", 
                     "warranty claim", "safety"]
if any(keyword in query.lower() for keyword in escalate_keywords):
    return "escalate"
```

---

## 🛠️ 调试技巧

### 1. 查看侧边栏信息
Streamlit 侧边栏会显示 3 个关键信息框：

#### 🔍 Retrieval Signals
```
- Top1: 0.5601
- Avg Top5: 0.5040
- Hits: 5
- Sections: TROUBLESHOOTING
```

#### 🔵 Router Decision
```
- Action: api
- Confidence: 0.85
- Reason: Query asks for live data (order status)
```

#### 🔬 RAG Debug Info（仅当 action=rag 时）
```
- can_answer: True
- confidence: 0.75
- status: success
```

### 2. 手动覆盖检索信号
在侧边栏启用 **"Manual Retrieval Signals"**，可以手动设置：
- Top1 Score
- Avg Top5 Score
- Hits Count

用于测试不同分数下的路由行为。

### 3. 命令行测试
```bash
# 测试 router 决策（不执行 action）
python utils/router_chain.py \
  --query "check my order status" \
  --top1 0.3 \
  --avg-top5 0.25 \
  --hits 4

# 预期输出：
# {"action": "api", "confidence": 0.85, "reason": "..."}
```

---

## 📝 测试清单

### API 路由
- [ ] "check my order ORD12345" → api
- [ ] "is the coffee machine in stock" → api
- [ ] "what is the current price" → api
- [ ] "check service status" → api

### RAG 路由
- [ ] "my coffee is not hot" → rag
- [ ] "how to clean the machine" → rag
- [ ] "what does the red light mean" → rag

### Escalate 路由
- [ ] "I want a refund" → escalate
- [ ] "connect me to support" → escalate
- [ ] "what's the weather" → escalate

### 边界情况
- [ ] 空查询 → escalate
- [ ] 非英文查询 → 根据意图
- [ ] 混合意图（"my order is broken"） → api（优先级高）

---

## 🔧 调整 Router 行为

如果 Router 判断不准确，可以调整：

### 1. 修改硬编码规则
编辑 `utils/router_chain.py` 第 134-141 行：
```python
# 降低 RAG 门槛
if top1 >= 0.4 and avg5 >= 0.30 and hits >= 3:
    return "rag"
    
# 提高 escalate 门槛
if top1 < 0.20 or avg5 < 0.15 or hits < 2:
    return "escalate"
```

### 2. 增强 API 关键词
编辑 `utils/router_chain.py` 第 72-100 行的 system_text，添加更多关键词。

### 3. 调整 LLM 温度
```python
# 更确定性的决策（默认）
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# 更多样化的决策
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
```

---

## 🎉 预期结果示例

### 成功的 API 路由
```
Query: "check my order ORD12345"

🔍 Retrieval Signals:
- Top1: 0.2301 (低分，因为不在知识库)
- Avg Top5: 0.1840
- Hits: 5

🔵 Router Decision:
- Action: api ✅
- Confidence: 0.85
- Reason: Query asks for live order data

📡 API Result:
{
  "order_id": "ORD12345",
  "status": "shipped",
  "tracking_number": "TRK123456789"
}
```

### 成功的 RAG 路由
```
Query: "my coffee is not hot"

🔍 Retrieval Signals:
- Top1: 0.5601 ✅
- Avg Top5: 0.5040 ✅
- Hits: 5 ✅

🔵 Router Decision:
- Action: rag ✅
- Confidence: 0.90
- Reason: High retrieval scores

🔬 RAG Debug:
- can_answer: True
- confidence: 0.82

✅ Answer: "To get hotter coffee, you can..."
```

---

**Happy Testing! 🚀**


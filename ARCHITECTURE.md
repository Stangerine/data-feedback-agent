# 数据回流 Agent 系统 — 单 Agent + Skills + Tools + 服务

## 一、设计思路

### 核心原则

1. **单 Agent**：一个 PI Agent 负责所有分析任务，通过 Skills 和 Tools 扩展能力
2. **服务化**：Python 分析能力封装为 FastAPI 服务，Agent 通过 HTTP 调用
3. **Skills 驱动**：每个分析维度一个 Skill，Agent 自动发现并按需加载
4. **渐进式**：先单 Agent 跑通全链路，后续任务复杂了再拆成 Multi-Agent

### 与 Multi-Agent 的关系

| | 当前架构（单 Agent） | 未来架构（Multi-Agent） |
|--|---------------------|----------------------|
| Agent 数量 | 1 个 | 每个维度一个 |
| 调度方式 | Agent 自己决定 | Coordinator 协调 |
| 工具调用 | 直接调用服务 | 通过 Coordinator 分发 |
| 故障隔离 | 工具级 | Agent 级 |
| 适用场景 | 任务明确、流程固定 | 任务复杂、需要并行 |

---

## 二、整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                          用户层 (User)                               │
│                                                                     │
│  输入: 线上图片目录路径 / 分析任务描述                                 │
│  输出: 回流决策报告 + 逐图分析详情                                    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     PI Agent 层 (单 Agent)                           │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                   数据回流 Agent                                │  │
│  │                                                               │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │  │
│  │  │ 检测验证     │ │ 分布分析     │ │ 质量评估     │            │  │
│  │  │ Skill       │ │ Skill       │ │ Skill       │            │  │
│  │  │             │ │             │ │             │            │  │
│  │  │ (判断标准)   │ │ (阈值规则)   │ │ (分级标准)   │            │  │
│  │  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘            │  │
│  │         │               │               │                    │  │
│  │  ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐            │  │
│  │  │ verify      │ │ embed       │ │ assess      │  ← Tools   │  │
│  │  │ batch_verify│ │ ood_detect  │ │ batch_assess│            │  │
│  │  │ health      │ │ class_shift │ │ quality_rpt │            │  │
│  │  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘            │  │
│  └─────────┼───────────────┼───────────────┼───────────────────┘  │
│            │               │               │                       │
│            │    HTTP REST  │  HTTP REST    │  HTTP REST            │
│            │               │               │                       │
└────────────┼───────────────┼───────────────┼───────────────────────┘
             │               │               │
             ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       服务层 (Python FastAPI)                         │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │ detection-   │  │ distribution-│  │ quality-     │             │
│  │ service      │  │ service      │  │ service      │             │
│  │ :8001        │  │ :8002        │  │ :8003        │             │
│  │              │  │              │  │              │             │
│  │ LLM 验证     │  │ CLIP 嵌入    │  │ LIQE 质量    │             │
│  │ YOLO 检测    │  │ 马氏距离 OOD │  │ 场景识别     │             │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘             │
│         │                 │                 │                       │
└─────────┼─────────────────┼─────────────────┼───────────────────────┘
          │                 │                 │
          ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     模型/数据层 (Model & Data)                        │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │  LLM API     │  │  CLIP 模型   │  │  LIQE 模型   │             │
│  │  Qwen3-VL    │  │  ViT-B/32    │  │  CLIP 微调    │             │
│  │  (Ollama)    │  │  512 维       │  │  质量/场景    │             │
│  └──────────────┘  └──────────────┘  └──────────────┘             │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │  YOLO 检测   │  │  训练集      │  │  Embedding   │             │
│  │  ONNX/Remote │  │  VOC 格式    │  │  缓存 .pkl   │             │
│  └──────────────┘  └──────────────┘  └──────────────┘             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│                         审计层 (Audit)                               │
│                                                                     │
│  行为审计: Agent 调用了哪些工具、输入输出是什么                         │
│  决策审计: 每张图片的评分过程和决策依据                                 │
│  数据血缘: 图片从输入到决策的完整链路                                  │
│                                                                     │
│  输出: output/audit/audit_log.jsonl                                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 三、Agent 配置

### 单 Agent 配置

```json
// .pi/settings.json
{
  "defaultProvider": "xiaomi",
  "defaultModel": "mimo-v2.5-pro",
  "defaultThinkingLevel": "medium"
}
```

### Tools 定义

| Tool | 说明 | 调用的服务 |
|------|------|-----------|
| `verify` | 验证单张图片的检测结果 | detection-service:8001 |
| `batch_verify` | 批量验证多张图片 | detection-service:8001 |
| `embed` | 提取图片 embedding | distribution-service:8002 |
| `ood_detect` | 检测 OOD 样本 | distribution-service:8002 |
| `class_shift` | 按类别分析分布偏移 | distribution-service:8002 |
| `assess` | 评估单张图片质量 | quality-service:8003 |
| `batch_assess` | 批量评估图片质量 | quality-service:8003 |
| `health_check` | 检查所有服务状态 | 各服务 |

---

## 四、Skills 设计

每个分析维度一个 Skill，定义判断标准和操作流程。Agent 在 system prompt 中看到 Skill 描述，需要时自动用 `read` 工具加载完整内容。

### Skill 1：检测验证 (`detection-review`)

```markdown
# 检测验证 Skill

## 职责
用多模态大模型验证小模型的检测结果，识别误报和漏报。

## 判断标准

### 误报 (False Positive)
- 检测框存在，但框内没有对应类别目标
- 检测框类别错误（如把 "chanche" 误检为 "dazhuangji"）
- 置信度低于 0.5 的检测框

### 漏报 (Missed Detection)
- 图片中存在目标但未检测到
- 目标被严重遮挡导致漏检（不算模型问题）
- 小目标（面积 < 32x32 像素）漏检（可接受）

### 评分规则
- 无误报无漏报: 1.0
- 有误报: 每个误报扣 0.1，最低 0.0
- 有漏报: 每个漏报扣 0.15，最低 0.0
- 误报+漏报同时存在: 0.0（直接不可信）

## 操作流程
1. 调用 `health_check` 确认 LLM 和 YOLO 服务可用
2. 调用 `verify` 或 `batch_verify` 验证图片
3. 根据结果判断图片是否适合回流
```

### Skill 2：分布分析 (`distribution-analysis`)

```markdown
# 分布分析 Skill

## 职责
分析线上图片与训练集的 embedding 分布差异，检测 OOD 样本。

## 判断标准

### OOD 检测
- 马氏距离 > 阈值（默认 3.0）: 标记为 OOD
- Mahalanobis distance > 5.0: 高度 OOD，优先回流

### 类别偏移
- 线上某类别占比 > 训练集占比的 2 倍: 该类别分布偏移
- 线上某类别占比 < 训练集占比的 0.5 倍: 该类别欠代表

### 评分规则
- 马氏距离 < 2.0: 1.0（分布一致）
- 2.0 <= 距离 < 3.0: 0.8（轻微偏移）
- 3.0 <= 距离 < 5.0: 0.5（明显偏移）
- 距离 >= 5.0: 0.2（严重偏移）

## 操作流程
1. 调用 `embed` 提取线上图片 embedding
2. 调用 `ood_detect` 检测 OOD 样本
3. 调用 `class_shift` 分析类别分布
4. 根据结果判断图片是否需要回流
```

### Skill 3：质量评估 (`quality-assessment`)

```markdown
# 质量评估 Skill

## 职责
评估图片视觉质量，分析模型在不同质量条件下的表现。

## 判断标准

### LIQE 质量分级
- 4.0 - 5.0: 优秀，适合训练
- 3.0 - 4.0: 良好，可接受
- 2.0 - 3.0: 一般，需谨慎使用
- < 2.0: 差，不建议回流

### 失真类型
- JPEG 压缩: 可接受
- 运动模糊: 视情况
- 低光照: 需评估
- 遮挡: 不影响（目标本身可能被遮挡）

### 评分规则
- LIQE >= 4.0: 1.0
- 3.0 <= LIQE < 4.0: 0.8
- 2.0 <= LIQE < 3.0: 0.5
- LIQE < 2.0: 0.2

## 操作流程
1. 调用 `assess` 或 `batch_assess` 评估图片质量
2. 根据 LIQE 分数和失真类型判断
3. 低质量图片不回流（避免引入噪声）
```

---

## 五、通信协议

### Agent → Service 调用

```
PI Agent (TypeScript)  ──HTTP──>  FastAPI Service (Python)
     │                                  │
     │  POST /api/verify                │
     │  { image_path, detections }      │
     │  ─────────────────────────────>  │
     │                                  │ ← 执行分析
     │  <─────────────────────────────  │
     │  { result: {...}, score: 0.85 }  │
```

### API 规范

所有服务遵循统一 API：

```yaml
# 健康检查
GET /health
Response: { "status": "ok", "service": "detection", "version": "1.0" }

# 分析接口
POST /api/analyze
Request: {
  "image_paths": ["path1.jpg"],
  "params": { ... }
}
Response: {
  "results": [
    {
      "image_path": "path1.jpg",
      "analysis": { ... },
      "score": 0.85,
      "processing_time_ms": 1200
    }
  ]
}
```

---

## 六、文件结构

```
agent_project/data-feedback-agent/
│
├── ARCHITECTURE.md                  ← 本文档
│
├── .pi/                             ← PI Agent 配置
│   ├── settings.json                ← 模型配置
│   ├── auth.json                    ← API Key
│   ├── models.json                  ← 自定义模型（Xiaomi MiMo）
│   ├── skills/                      ← 技能定义
│   │   ├── detection-review/
│   │   │   └── SKILL.md
│   │   ├── distribution-analysis/
│   │   │   └── SKILL.md
│   │   └── quality-assessment/
│   │       └── SKILL.md
│   └── extensions/
│       ├── audit-log.ts             ← 审计扩展
│       └── service-health.ts        ← 服务健康检查
│
├── tools/                           ← PI Agent 工具封装
│   ├── detection-tool.ts            ← 调用 detection-service
│   ├── distribution-tool.ts         ← 调用 distribution-service
│   ├── quality-tool.ts              ← 调用 quality-service
│   └── tools.ts                     ← 工具注册入口
│
├── services/                        ← Python FastAPI 服务
│   ├── detection-service/
│   │   ├── app.py                   ← FastAPI 入口
│   │   ├── verifier.py              ← LLM 验证逻辑
│   │   ├── detector.py              ← YOLO 检测逻辑
│   │   ├── prompts.py               ← Prompt 模板
│   │   └── requirements.txt
│   │
│   ├── distribution-service/
│   │   ├── app.py
│   │   ├── embedder.py              ← CLIP embedding
│   │   ├── ood_detector.py          ← OOD 检测
│   │   ├── cache/                   ← 训练集 embedding 缓存
│   │   └── requirements.txt
│   │
│   └── quality-service/
│       ├── app.py
│       ├── assessor.py              ← LIQE 评估
│       ├── models/LIQE.pt
│       └── requirements.txt
│
├── config/
│   ├── services.yaml                ← 服务地址和端口
│   └── scoring.yaml                 ← 评分权重和阈值
│
├── scripts/
│   ├── start_services.sh            ← 启动所有 Python 服务
│   ├── stop_services.sh             ← 停止所有服务
│   └── run_analysis.sh              ← 一键运行分析
│
├── output/                          ← 分析结果
│   ├── audit/                       ← 审计日志
│   ├── per_image/                   ← 逐图分析详情
│   ├── feedback_list.json           ← 回流候选列表
│   └── summary.json                 ← 汇总报告
│
└── agent.ts                         ← Agent 入口脚本
```

---

## 七、权限矩阵

| Tool | 读图片 | 读训练集 | 调用 LLM | 调用 YOLO | 读检测结果 | 写结果 |
|------|--------|---------|---------|----------|-----------|--------|
| verify | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| batch_verify | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| embed | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ |
| ood_detect | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ |
| class_shift | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ |
| assess | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| batch_assess | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |

---

## 八、审计系统

### 审计目标

1. **谁做了什么** — Agent 调用了哪些工具，输入输出是什么
2. **为什么这样决策** — 每张图片的评分过程和决策依据
3. **数据从哪来** — 图片从输入到决策的完整链路

### 审计日志格式

```json
{
  "timestamp": "2026-06-11T10:30:00Z",
  "event_type": "tool_call | tool_result | decision | error",
  "session_id": "sess_abc123",
  "image_id": "img_001.jpg",
  "tool": "verify",
  "input": { "image_path": "...", "detections": [...] },
  "output": { "false_positive": false, "missed_detection": true },
  "duration_ms": 1200,
  "score": 0.85,
  "metadata": { "model": "qwen3-vl:8b" }
}
```

### 实现方式

PI Agent 扩展（`audit-log.ts`）：

```typescript
function auditLogExtension(pi: ExtensionAPI) {
  const logFile = resolve(projectDir, "output/audit/audit_log.jsonl");

  pi.on("tool_call", async (event) => {
    const record = {
      timestamp: new Date().toISOString(),
      event_type: "tool_call",
      session_id: pi.getSessionId(),
      tool: event.toolName,
      input: event.input,
    };
    appendFileSync(logFile, JSON.stringify(record) + "\n");
  });

  pi.on("tool_execution_end", async (event) => {
    const record = {
      timestamp: new Date().toISOString(),
      event_type: "tool_result",
      tool: event.toolName,
      output_summary: summarizeOutput(event.result),
      duration_ms: event.duration,
      success: !event.isError,
    };
    appendFileSync(logFile, JSON.stringify(record) + "\n");
  });
}
```

---

## 九、数据流

### 完整分析流程

```
用户输入：线上图片目录路径
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│  PI Agent                                                │
│                                                          │
│  1. 扫描目录，列出所有图片                                 │
│  2. 按 Skill 依次执行分析                                 │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Step 1: 检测验证                                    │ │
│  │  加载 detection-review Skill                         │ │
│  │  调用 verify → detection-service:8001               │ │
│  │  → LLM 验证 + YOLO 检测                             │ │
│  │  → 输出: 误报/漏报信息 + 检测分数                     │ │
│  └────────────────────────────────────────────────────┘ │
│                          │                               │
│                          ▼                               │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Step 2: 分布分析                                    │ │
│  │  加载 distribution-analysis Skill                   │ │
│  │  调用 embed → distribution-service:8002             │ │
│  │  → CLIP 嵌入 + 马氏距离 OOD                         │ │
│  │  → 输出: OOD 标记 + 分布偏移分数                      │ │
│  └────────────────────────────────────────────────────┘ │
│                          │                               │
│                          ▼                               │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Step 3: 质量评估                                    │ │
│  │  加载 quality-assessment Skill                      │ │
│  │  调用 assess → quality-service:8003                 │ │
│  │  → LIQE 质量评分                                    │ │
│  │  → 输出: 质量分数 + 失真类型                          │ │
│  └────────────────────────────────────────────────────┘ │
│                          │                               │
│                          ▼                               │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Step 4: 融合决策                                    │ │
│  │                                                      │ │
│  │  回流分数 = 0.35×检测 + 0.30×分布 + 0.15×质量        │ │
│  │                                                      │ │
│  │  决策规则:                                            │ │
│  │  - 分数 >= 0.7: 高优先级回流                          │ │
│  │  - 0.4 <= 分数 < 0.7: 低优先级回流                   │ │
│  │  - 分数 < 0.4: 不回流                                │ │
│  └────────────────────────────────────────────────────┘ │
│                          │                               │
│                          ▼                               │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Step 5: 输出结果                                    │ │
│  │  - feedback_list.json（回流候选）                     │ │
│  │  - summary.json（汇总报告）                          │ │
│  │  - per_image/*.json（逐图详情）                      │ │
│  │  - audit/audit_log.jsonl（审计日志）                 │ │
│  └────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### 单张图片的完整链路

```
img_001.jpg
  │
  ├─→ 检测验证: verify({image_path, detections})
  │   → {误报: false, 漏报: true, score: 0.7}
  │
  ├─→ 分布分析: embed({image_path}) → ood_detect({embedding})
  │   → {ood: false, distance: 2.5, score: 0.8}
  │
  ├─→ 质量评估: assess({image_path})
  │   → {liqe: 3.8, scene: cityscape, score: 0.8}
  │
  └─→ 融合决策:
      回流分数 = 0.35×0.7 + 0.30×0.8 + 0.15×0.8 = 0.765
      决策: 高优先级回流
```

---

## 十、服务启停管理

### 启动顺序

```bash
# 1. 启动 Python 服务
./scripts/start_services.sh

# 2. 启动 PI Agent（服务就绪后）
npx pi run agent.ts
```

### 服务配置

```yaml
# config/services.yaml
services:
  detection:
    url: http://localhost:8001
    health_check: /health
    timeout: 300
  distribution:
    url: http://localhost:8002
    health_check: /health
    timeout: 60
  quality:
    url: http://localhost:8003
    health_check: /health
    timeout: 30
```

---

## 十一、实现路线

### Phase 1：服务化基础（2 周）

1. 将 `LLM_verify/verify.py` 封装为 `detection-service` FastAPI 服务
2. 将 LIQE 封装为 `quality-service` FastAPI 服务
3. 实现 PI Agent 工具封装（HTTP 调用服务）
4. 验证 Agent → Service 的调用链路

### Phase 2：分布分析服务（1 周）

1. 实现 `distribution-service`（CLIP + 马氏距离）
2. 实现训练集 embedding 缓存
3. 完成 3 个服务的独立运行

### Phase 3：Skills 和审计（1 周）

1. 编写 3 个 Skills（检测验证、分布分析、质量评估）
2. 实现审计扩展（audit-log.ts）
3. 实现服务健康检查扩展

### Phase 4：生产化（1 周）

1. 配置化（阈值、权重可调）
2. 容错处理（服务不可用时的降级策略）
3. Docker 化部署
4. 完善文档和使用示例

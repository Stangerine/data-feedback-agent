# 数据回流 Agent 系统 — 项目详解

> 面向目标检测 bad case（误报/漏报）的数据回流闭环系统，用于工地安检场景中施工车辆检测模型的持续迭代优化。

---

## 一、项目背景与核心问题

### 1.1 业务场景

工地安检场景下，使用 YOLO 小模型对 9 类施工车辆（挖掘机、铲车、打桩机、压路机、吊车、高空车、油罐车、运输车、其他）进行实时检测。小模型在线上部署后会出现两类典型 bad case：

- **误报（False Positive）**：模型将非目标物体错误识别为车辆，或类别判断错误（如把铲车误检为打桩机）
- **漏报（Missed Detection / False Negative）**：图片中存在车辆但模型未检测到

### 1.2 核心挑战

传统人工标注回流效率低、成本高。需要一个**自动化 Agent 系统**来：
1. 自动识别哪些样本是误报/漏报
2. 分析误报/漏报的根因（是光照问题？遮挡？还是类别混淆？）
3. 自动校准检测框（修正类别、补充漏检框）
4. 筛选出高价值回流样本，支撑模型持续迭代

### 1.3 最终效果

完成一轮数据回流闭环迭代后：
- 漏报/误报识别**召回率达到 83.6%**
- 目标检测小模型 **mAP 提升 1.7 个百分点**

---

## 二、系统架构设计

### 2.1 整体架构：单 Agent + Skills + Tools + 微服务

采用**两层架构**：TypeScript PI Agent 负责任务编排和用户交互，Python FastAPI 微服务负责具体计算。

```
用户（自然语言）
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  PI Agent 层（TypeScript, GPT-5.5 推理）              │
│                                                     │
│  3 个 Skills（定义工作流）                              │
│    ├── detection-review   检测验证                    │
│    ├── detection-correction 检测校准                  │
│    └── data-analysis      多维归因分析                │
│                                                     │
│  7 个 Tools（HTTP 调用封装）                           │
│    ├── 4 个检测工具 → detection-service (:8001)       │
│    └── 3 个分析工具 → data-analysis-service (:8002)   │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP REST
         ┌─────────────┴─────────────┐
         ▼                           ▼
┌──────────────────┐      ┌──────────────────────┐
│ detection-service │      │ data-analysis-service │
│ :8001             │      │ :8002                 │
│                   │      │                       │
│ • YOLO 远程检测    │      │ • 6 维语义分析         │
│ • LLM 误报/漏报验证│      │   (BGE-VL-large)      │
│ • LLM 校准修正     │      │ • 类别分布分析          │
│ • 可视化输出       │      │ • LLM 归因分析          │
│                   │      │ • 综合评分排序          │
│ 调用: YOLO API     │      │ 调用: BGE-VL + MiMo    │
│       MiMo v2.5   │      │       (CUDA GPU)       │
└──────────────────┘      └──────────────────────┘
```

### 2.2 为什么选择这个架构

| 设计决策 | 原因 |
|---------|------|
| 单 Agent 而非 Multi-Agent | 当前任务流程固定，单 Agent 足够；后续任务复杂了可拆分为多 Agent |
| Agent 用 GPT-5.5，服务用 MiMo v2.5 | Agent 负责推理和工具选择（需要强推理能力），服务做结构化任务（MiMo 更快更便宜） |
| Skill 驱动工作流 | 每个 Skill 是一个标准化流程定义，Agent 按需加载，保证流程一致性 |
| Function Calling 而非纯文本解析 | 用 OpenAI 风格的 function calling 让 LLM 输出结构化 JSON，配合 JSON fallback 解析保证鲁棒性 |

---

## 三、核心流程一：问题样本判断（误报/漏报识别）

### 3.1 目标

对小模型的检测结果进行**二次校验**，自动识别误报和漏报。

### 3.2 流程

```
输入：一张图片 + 小模型检测结果（bbox 列表）
                │
                ▼
    ┌───────────────────────┐
    │ Step 1: YOLO 远程检测   │  ← 可选，如果已有检测结果可跳过
    │ DetectionClient.detect │
    │ 调用远程 YOLO API       │
    │ 返回 bbox 列表          │
    └───────────┬───────────┘
                │
                ▼
    ┌───────────────────────┐
    │ Step 2: 构建验证 Prompt  │
    │                        │
    │ System Prompt 包含：    │
    │ • 9类车辆定义及中文名    │
    │ • 各类车辆视觉特征描述   │
    │ • 误报判断标准          │
    │ • 漏报判断标准          │
    │ • 互斥规则             │
    │ • 置信度要求           │
    │                        │
    │ User Prompt 包含：      │
    │ • 图片（base64）        │
    │ • 检测框列表            │
    │ • 每个框的类别+置信度    │
    └───────────┬───────────┘
                │
                ▼
    ┌───────────────────────┐
    │ Step 3: LLM Function   │
    │ Calling 验证            │
    │                        │
    │ 调用 MiMo v2.5          │
    │ 使用 report_verification│
    │ function calling tool   │
    │                        │
    │ 输出结构化 JSON：       │
    │ • false_positives[]    │
    │ • missed_detections[]  │
    │ • overall_assessment   │
    └───────────┬───────────┘
                │
                ▼
    ┌───────────────────────┐
    │ Step 4: 结果解析 & 重试  │
    │                        │
    │ 优先路径：从 tool_calls │
    │ 提取结构化参数           │
    │                        │
    │ Fallback：从文本中解析   │
    │ JSON（正则匹配）        │
    │                        │
    │ 失败时重试最多 2 次      │
    │ 每次追加上次错误信息     │
    └───────────┬───────────┘
                │
                ▼
    ┌───────────────────────┐
    │ Step 5: 审计记录        │
    │ 保存 JSON 审计文件       │
    │ 含：时间、图片、检测结果、│
    │ 原始输出、解析结果、      │
    │ 尝试次数、耗时           │
    └───────────────────────┘
```

### 3.3 Prompt 设计要点

**误报判断标准**：
- 检测框内没有对应类别目标 → 误报
- 类别判断错误（如铲车误检为打桩机）→ 误报
- IoU < 0.3 的检测框 → 误报

**漏报判断标准**（保守策略，防止误判）：
- 目标必须**清晰可见**（非严重遮挡）
- 目标必须**足够大**（非极小目标）
- 目标必须**可识别类别**（不能是模糊物体）
- 图中不能已有相同位置的检测框覆盖

**互斥规则**：同一个目标不能同时被标记为误报和漏报

**置信度要求**：每个误报和漏报都必须给出 0-1 的数值置信度

### 3.4 Function Calling Schema

```python
# report_verification 工具定义
{
    "false_positives": [{
        "detection_index": 0,        # 对应检测结果序号
        "reported_class": "dazhuangji", # 小模型给出的类别
        "actual_class": "chanche",     # 大模型判断的实际类别
        "confidence": 0.95,           # 置信度
        "reason": "该设备带有铲斗..."  # 判断理由
    }],
    "missed_detections": [{
        "actual_class": "wajueji",    # 实际类别
        "actual_class_cn": "挖掘机",
        "confidence": 0.88,
        "region_hint": {              # 位置提示（归一化坐标）
            "x1": 0.3, "y1": 0.2,
            "x2": 0.7, "y2": 0.6
        },
        "location": "画面右侧",
        "description": "一台黄色挖掘机...",
        "confidence_level": "high"    # high/medium/low
    }],
    "overall_assessment": {
        "total_detections": 3,
        "fp_count": 1,
        "md_count": 1,
        "detection_quality": "fair",  # good/fair/poor
        "summary": "存在1个类别误判和1个漏检..."
    }
}
```

---

## 四、核心流程二：多维归因分析

### 4.1 目标

对误报/漏报样本进行**根因分析**，从多个维度分析问题原因，为后续校准和回流决策提供依据。

### 4.2 6 个语义分析维度

使用 **BGE-VL-large**（CLIP 变体，本地 CUDA GPU）进行图像-文本语义匹配：

| 维度 | 类别 | 分析方法 |
|------|------|---------|
| **光照 (lighting)** | bright / moderate / dim | 图像 embedding 与文本 prompt 余弦相似度 |
| **视角 (viewpoint)** | front / side / rear / overhead | 同上 |
| **清晰度 (blur)** | sharp / motion-blur / out-of-focus | 同上 |
| **天气 (weather)** | clear / cloudy / rain / snow / fog | 同上 |
| **时段 (timeOfDay)** | day / dusk / night | 同上 |
| **环境 (environment)** | indoor / urban-street / construction-site / rural-field / aerial-scene | 同上 |

**分析原理**：

```
输入图片 → BGE-VL-large → 图像 embedding (512维)
                              │
                              ▼
                    与各维度的文本 prompt 计算余弦相似度
                    例如光照维度：
                      "a bright well lit image"  → sim₁
                      "a moderately lit image"   → sim₂
                      "a dim dark image"         → sim₃
                              │
                              ▼
                    softmax(sim₁, sim₂, sim₃) → 概率分布
                    最大概率的类别 = 该图片的光照分类
```

### 4.3 类别分布分析（ClassAnalyzer）

对比训练集与测试集的类别分布：

- **覆盖率缺口（Coverage Gap）**：训练集中某类占比 < 2% → 标记为稀有类
- **过代表类（Overrepresented）**：测试集中某类占比 > 训练集占比 × 1.5 → 标记为偏高
- **回流价值评分**：`0.6 × rarity_score + 0.4 × prevalence_score`

### 4.4 预计算机制

服务启动时**预计算训练集分布**，避免每次请求重复计算：

```
服务启动
    │
    ▼
加载训练集标注 → ClassAnalyzer 计算类别分布
    │
    ▼
遍历训练集图片 → 6 个 SemanticAnalyzer 批量计算
    │              生成 train_semantic_distribution
    ▼
缓存在内存中，后续每张测试图片只需计算自身特征，与缓存分布对比
```

### 4.5 LLM 归因分析（LLMAttributionAnalyzer）

将 6 维分析结果汇总，由 LLM 进行最终归因：

**输入给 LLM 的信息**：
- 图片路径和检测结果
- 问题类型（误报/漏报）
- 类别覆盖率分析（缺口 + 过代表类）
- 6 个语义维度的分析结果 + 与训练集分布的对比

**LLM 输出结构**：

```python
{
    "attribution_type": "environment",     # 归因类型
    # 可选值：background_noise / class_confusion / occlusion
    #        environment / annotation_error / other
    "confidence": 0.85,
    "reasoning": "该图片为夜间施工场景，训练集中夜间样本占比仅3%...",
    "dimension_attributions": [{
        "dimension": "lighting",
        "contribution": 0.4,        # 该维度对问题的贡献度
        "reasoning": "夜间低光照导致车辆轮廓不清晰"
    }, ...],
    "main_cause_dimension": "lighting",
    "feedback_suggestion": "建议补充夜间施工场景的标注数据",
    "should_feedback": true            # 是否建议回流
}
```

### 4.6 归因类型说明

| 归因类型 | 含义 | 回流价值 |
|---------|------|---------|
| background_noise | 背景干扰，模型将背景误识别为目标 | 中 |
| class_confusion | 类间混淆，相似车型互相误判 | 高 |
| occlusion | 遮挡导致漏检或误判 | 高 |
| environment | 环境因素（光照、天气等） | 最高 |
| annotation_error | 标注错误（原标注就有问题） | 需人工复核 |
| other | 其他原因 | 视情况 |

---

## 五、核心流程三：自动校准

### 5.1 目标

对已确认的误报和漏报进行**自动修正**，生成校准后的检测结果和可视化对比图。

### 5.2 校准流程

```
输入：验证结果（误报列表 + 漏报列表）+ 原始检测框
                │
                ▼
    ┌───────────────────────┐
    │ 误报校准（类别修正）     │
    │                        │
    │ 如果误报是类别错误：     │
    │   reported_class →      │
    │   actual_class          │
    │   source = "llm_corrected"│
    │                        │
    │ 如果误报是纯虚检：       │
    │   直接移除该检测框       │
    └───────────┬───────────┘
                │
                ▼
    ┌───────────────────────┐
    │ 漏报校准（补充 bbox）    │
    │                        │
    │ 构建校准 Prompt：       │
    │ • 验证结果作为唯一真相   │
    │ • 漏报的 region_hint     │
    │ • 车辆视觉特征参考       │
    │                        │
    │ LLM Function Calling   │
    │ 输出精确的 bbox 坐标     │
    │ （归一化 xyxy 格式）     │
    └───────────┬───────────┘
                │
                ▼
    ┌───────────────────────┐
    │ 质量过滤                │
    │                        │
    │ 过滤条件：              │
    │ • 置信度 < 0.5 → 丢弃  │
    │ • 面积比 < 0.1% → 丢弃 │
    │ • 与已有框 IoU > 0.5   │
    │   → 丢弃（去重）        │
    │                        │
    │ 通过过滤的标记为         │
    │ source = "llm_added"   │
    └───────────┬───────────┘
                │
                ▼
    ┌───────────────────────┐
    │ 可视化输出              │
    │                        │
    │ small_model.jpg：       │
    │   蓝色框 = 小模型原始结果│
    │                        │
    │ corrected.jpg：         │
    │   蓝色 = 未修改的框     │
    │   橙色 = 类别修正的框   │
    │   红色 = LLM 新增的框   │
    │                        │
    │ result.json：完整结果   │
    └───────────────────────┘
```

### 5.3 关键设计："验证结果是唯一真相"

校准阶段的 LLM Prompt 明确约束：**验证结果是唯一真相（single source of truth）**。校准 LLM 不允许重新判断误报/漏报，只负责为已确认的漏报提供精确的 bbox 坐标。这避免了校准阶段引入新的判断错误。

### 5.4 校准后的检测结果格式

```python
{
    "detections": [
        {
            "bbox": [x1, y1, x2, y2],   # 像素坐标
            "class_name": "wajueji",
            "class_id": 0,
            "confidence": 0.92,
            "source": "small_model"       # 未修改
        },
        {
            "bbox": [x1, y1, x2, y2],
            "class_name": "chanche",       # 原来误检为 dazhuangji
            "class_id": 1,
            "confidence": 0.95,
            "source": "llm_corrected"     # 类别修正
        },
        {
            "bbox": [x1, y1, x2, y2],
            "class_name": "wajueji",
            "class_id": 0,
            "confidence": 0.88,
            "source": "llm_added"         # LLM 新增
        }
    ]
}
```

---

## 六、核心流程四：回流建议生成（综合评分）

### 6.1 评分公式

```
feedback_score = 0.25 × class_score        # 类别覆盖分
               + 0.30 × embedding_score     # Embedding 分布分
               + 0.15 × quality_score       # 图像质量分
               + 0.15 × spatial_score       # 空间特征分
               + 0.15 × llm_score           # LLM 归因分
```

### 6.2 各维度评分说明

**类别覆盖分 (0.25)**：
- 该样本属于训练集稀有类 → 高分
- 该样本属于训练集已充分覆盖的类 → 低分

**Embedding 分布分 (0.30)**（权重最高）：
- 测试图片 embedding 与训练集分布的距离（OOD 分数）
- OOD 分数越高 → 与训练集差异越大 → 回流价值越高
- 使用 BGE-VL-large 提取 embedding，计算余弦距离

**图像质量分 (0.15)**：
- 基于语义分析的清晰度、光照等维度综合评估
- 质量过低的图片不应回流（会引入噪声）

**空间特征分 (0.15)**：
- 小目标（面积 < 0.5%）和边缘目标占比
- 小目标/边缘目标多 → 模型更难识别 → 回流价值高

**LLM 归因分 (0.15)**：
- LLM 判断的归因类型
- 环境因素和遮挡导致的误报 → 回流价值最高
- 背景干扰 → 中等价值
- 标注错误 → 需人工复核

### 6.3 回流决策阈值

| 评分区间 | 决策 | 说明 |
|---------|------|------|
| >= 0.6 | **推荐回流** (high_value) | 高价值样本，优先加入训练集 |
| 0.3 ~ 0.6 | **建议人工复核** (medium_value) | 有一定价值，需人工确认 |
| < 0.3 | **不建议回流** (low_value) | 低价值或可能引入噪声 |

---

## 七、完整数据流（端到端）

以一张图片为例，完整链路：

```
用户: "帮我分析这张图片的检测结果"
        │
        ▼
  PI Agent（GPT-5.5 推理，选择工具）
        │
        ├─→ 1. verify_detection（检测验证）
        │     → detection-service
        │     → YOLO 检测 + MiMo LLM 验证
        │     → 返回：1个误报（类别混淆），1个漏报
        │
        ├─→ 2. compare_data（多维归因分析）
        │     → data-analysis-service
        │     → 6 维语义分析：夜间(dim) + 侧视(side) + 清晰(sharp)
        │     →            + 阴天(cloudy) + 夜间(night) + 工地(construction-site)
        │     → 类别分析：漏报的挖掘机属于稀有类（训练集仅 2.3%）
        │     → LLM 归因：主要原因是夜间低光照（贡献度 0.4）
        │     → 综合评分：0.72 → 推荐回流
        │
        └─→ 3. correct_detection（自动校准）
              → detection-service
              → 误报修正：dazhuangji → chanche（类别修正）
              → 漏报补充：LLM 输出精确 bbox
              → 质量过滤后生成 corrected.jpg
              → 保存 small_model.jpg + corrected.jpg + result.json
```

---

## 八、技术实现细节

### 8.1 LLM 客户端抽象（工厂模式）

```python
# 支持三种 LLM 后端
create_llm_client("openai")    # OpenAI 兼容 API（MiMo v2.5 用这个）
create_llm_client("anthropic") # Anthropic Claude
create_llm_client("ollama")    # Ollama 本地模型
```

所有客户端继承 `BaseLLMClient`，统一接口：`chat(system_prompt, user_prompt, image, tools)`

### 8.2 配置管理

所有服务共享项目根目录的 `config.yaml`，通过**向上遍历目录树**的方式查找。支持环境变量覆盖关键字段。

### 8.3 Embedding 缓存

BGE-VL-large 的图像 embedding 计算较慢，采用**文件缓存**策略：
- 缓存 key：图片文件的 MD5 哈希
- 缓存格式：numpy `.npy` 文件
- 缓存目录：`./semantic_cache/`
- 同一张图片不会重复计算

### 8.4 审计系统

每次验证尝试都保存 JSON 审计记录：
```
audit/
  2026-06-25/
    verify_20260625_103000_img001.json
    verify_20260625_103002_img001.json   ← 重试记录
```

记录内容：时间戳、图片路径、检测结果、LLM 原始输出、解析结果、尝试次数、耗时、错误信息。

---

## 九、项目亮点（面试回答要点）

### 1. Function Calling 保证结构化输出

不是让 LLM 自由生成文本再用正则解析，而是通过 OpenAI 风格的 function calling 定义严格的 JSON Schema，让 LLM 按 schema 输出。配合 fallback 解析（正则匹配 `{...}`），保证了 99%+ 的解析成功率。

### 2. 保守的漏报判断策略

Prompt 中设计了多层约束：只报告"清晰可见 + 足够大 + 可识别类别"的目标，且不能与已有检测框重叠。这避免了 LLM 过度报告漏报导致的数据污染。

### 3. 验证-校准两阶段分离

验证阶段只做判断（误报/漏报），校准阶段只做修正（输出 bbox）。校准 LLM 被明确约束为"验证结果是唯一真相"，不允许重新判断，避免错误传播。

### 4. 预计算 + 缓存的性能优化

训练集分布（类别分布 + 6 维语义分布）在服务启动时预计算，单张图片分析只需计算自身特征并与缓存对比。Embedding 使用 MD5-keyed 文件缓存，避免重复计算。

### 5. 多维度归因而非单一判断

不是简单地判断"这个样本值不值得回流"，而是从类别覆盖、语义分布、图像质量、空间特征、LLM 归因 5 个维度综合分析，给出可解释的归因结果和回流建议。

### 6. Agent 编排 + 微服务执行的分层设计

Agent 层（GPT-5.5）负责理解用户意图和工具选择，服务层（MiMo v2.5）负责具体计算。两层使用不同的 LLM，各取所长：Agent 需要强推理能力，服务需要快速稳定的结构化输出。

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
│    ├── detection-review     检测校验（只分析不生成图）  │
│    ├── detection-correction 检测纠正（校验+可视化）     │
│    └── data-analysis        误报归因与数据回流分析      │
│                                                     │
│  9 个 Tools（HTTP 调用封装）                           │
│    ├── 4 个检测工具 → detection-service (:8001)       │
│    └── 5 个分析工具 → data-analysis-service (:8002)   │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP REST
         ┌─────────────┴─────────────┐
         ▼                           ▼
┌──────────────────┐      ┌──────────────────────┐
│ detection-service │      │ data-analysis-service │
│ :8001             │      │ :8002                 │
│                   │      │                       │
│ • YOLO 远程检测    │      │ • 7 维语义分析         │
│ • LLM 误报/漏报验证│      │   (BGE-VL-large)      │
│ • LLM 校准修正     │      │ • 类别分布分析          │
│ • 可视化输出       │      │ • LLM 归因分析          │
│                   │      │ • 批量分析与导出        │
│ 调用: YOLO API     │      │ 调用: BGE-VL + MiMo    │
│       MiMo v2.5   │      │       (CUDA GPU)       │
└──────────────────┘      └──────────────────────┘
```

### 2.2 为什么选择这个架构

| 设计决策 | 原因 |
|---------|------|
| 单 Agent 而非 Multi-Agent | 当前任务流程固定，单 Agent 足够；后续任务复杂了可拆分为多 Agent |
| Agent 用 GPT-5.5，服务用 MiMo v2.5 | Agent 负责推理和工具选择（需要强推理能力），服务做结构化任务（MiMo 更快更便宜）。两者都通过 OpenAI 兼容 API 调用 |
| Skill 驱动工作流 | 每个 Skill 是一个标准化流程定义，Agent 按需加载，保证流程一致性 |
| Function Calling 而非纯文本解析 | 用 OpenAI 风格的 function calling 让 LLM 输出结构化 JSON，配合 JSON fallback 解析保证鲁棒性 |

### 2.3 工具清单（9 个）

**检测工具（→ detection-service :8001）**：

| 工具名 | 功能 | 调用的 API |
|--------|------|-----------|
| `verify_detection` | 检测 + LLM 校验 | `/api/verify` |
| `correct_detection` | 检测 + LLM 纠正 + 可视化 | `/api/correct` |
| `verify_direct` | 直接校验（跳过检测 API） | `/api/verify_direct` |
| `check_detection_service` | 检测服务健康检查 | `/health` |

**分析工具（→ data-analysis-service :8002）**：

| 工具名 | 功能 | 调用的 API |
|--------|------|-----------|
| `analyze_dataset` | 数据集画像（类别分布、bbox 统计） | `/api/profile` |
| `analyze_single` | 单图 7 维归因分析 | `/api/analyze/single` |
| `analyze_batch` | 批量归因分析 | `/api/analyze/batch` |
| `export_analysis` | 导出分析结果（JSON/CSV） | `/api/export` |
| `check_analysis_service` | 分析服务健康检查 | `/health` |

### 2.4 Skill 清单（3 个）

| Skill | 触发场景 | 使用的工具 |
|-------|---------|-----------|
| `detection-review` | 用户要求校验检测结果、检查误报 | verify_detection, verify_direct, check_detection_service |
| `detection-correction` | 用户要求纠正检测、生成对比图 | correct_detection, check_detection_service |
| `data-analysis` | 用户要求分析误报原因、回流建议 | analyze_single, analyze_batch, analyze_dataset, export_analysis |

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

### 4.2 7 个分析维度

分析分为 7 个维度，其中 6 个使用 **BGE-VL-large**（CLIP 变体，本地 CUDA GPU）进行图像-文本语义匹配，1 个基于统计分析：

| 维度 | 类别 | 分析方法 |
|------|------|---------|
| **类别 (class)** | 9 类施工车辆 | 训练集类别分布统计，识别稀有类 |
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
- **类别覆盖评分**：稀有类的误报/漏报更容易被判定为"数据缺陷"，建议优先回流

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
    │              (lighting/viewpoint/blur/weather/timeOfDay/environment)
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
- 7 个分析维度的结果 + 与训练集分布的对比

**LLM 输出结构**：

```python
{
    "filename": "20250304175558461.jpg",
    "attribution_type": "类别偏差",     # 归因类型
    # 可选值：光照问题 / 视角问题 / 清晰度问题 / 天气问题 / 时段问题 / 环境问题 / 类别偏差 / 类间混淆 / 其他
    "confidence": 0.90,
    "reasoning": "打桩机在训练集中仅占1.1%，样本严重不足...",
    "dimension_attributions": [{
        "dimension": "class",
        "category": "dazhuangji",
        "train_coverage": 0.011,        # 训练集中该类别的占比
        "is_gap": true,                 # 是否是覆盖缺口
        "contribution": "打桩机样本严重不足，模型缺乏泛化能力"
    }, ...],
    "main_cause_dimension": "class",
    "feedback_suggestion": "应优先补充打桩机(dazhuangji)相关训练数据",
    "should_feedback": true             # 是否建议回流
}
```

### 4.6 归因类型说明

| 归因类型 | 对应维度 | 含义 | 回流价值 |
|---------|----------|------|---------|
| 光照问题 | lighting | 光照条件（过暗/过亮/逆光）导致误报漏报 | 高 |
| 视角问题 | viewpoint | 拍摄角度（仰角/俯拍/侧面）差异导致 | 高 |
| 清晰度问题 | blur | 运动模糊/失焦/低分辨率导致 | 高 |
| 天气问题 | weather | 雨雪雾等特殊天气条件导致 | 中 |
| 时段问题 | timeOfDay | 夜间/黄昏等时段差异导致 | 中 |
| 环境问题 | environment | 施工场景/城市街道/室内等环境差异 | 最高 |
| 类别偏差 | class | 训练集该类样本不足，分布不均 | **最高** |
| 类间混淆 | — | 不同类别外观相似导致混淆（如吊车与打桩机） | 高 |
| 其他 | — | 上述维度外的其他原因 | 视情况 |

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

## 六、核心流程四：回流建议生成（LLM 归因决策）

### 6.1 决策机制

当前版本不再使用加权评分公式，而是由 **LLM 综合所有维度信息直接做出归因判断和回流建议**。LLM 会：

1. 查看该图片在 7 个维度上的分类结果
2. 对比训练集中各维度的覆盖情况
3. 综合判断误报/漏报的根本原因
4. 给出是否建议回流的二元决策 + 中文建议

### 6.2 回流决策逻辑

| 判断依据 | 决策 | 说明 |
|---------|------|------|
| 训练集缺少某维度场景 | **建议回流** | 类别偏差/环境问题是强回流信号 |
| 多个维度存在覆盖缺口 | **建议回流** | 综合覆盖不足 |
| 仅单一时段/天气因素导致 | **视情况** | 如果该条件频繁出现则回流 |
| 类间混淆导致 | **建议回流** | 补充难例样本 |
| 所有维度覆盖充足 | **不建议回流** | 可能是模型本身的问题 |

### 6.3 归因置信度

LLM 同时输出归因置信度 (0-1)：
- **>= 0.8**: 高确信，归因结果可靠
- **0.5 ~ 0.8**: 中等确信，建议结合人工判断
- **< 0.5**: 低确信，需要更多样本验证

---

## 七、完整数据流（端到端）

以一张图片为例，完整链路：

```
用户: "帮我分析这张图片的检测结果"
        │
        ▼
  PI Agent（GPT-5.5 推理，选择工具）
        │
        ├─→ 1. verify_detection（检测校验）
        │     → detection-service
        │     → YOLO 检测 + MiMo LLM 校验
        │     → 返回：1个误报（吊车→打桩机），1个漏报（吊车）
        │
        ├─→ 2. analyze_single（单图归因分析）
        │     → data-analysis-service
        │     → 7 维分析：
        │       class: dazhuangji, 训练集仅 1.1% ← 覆盖缺口
        │       lighting: dim (89.8%)
        │       viewpoint: overhead (31.2%)
        │       blur: motion-blur (86.1%)
        │       weather: rain (52.4%)
        │       timeOfDay: dusk (15.9%)
        │       environment: construction-site (96.0%)
        │     → LLM 归因：数据缺陷，打桩机样本严重不足
        │     → 置信度：0.90 → 建议回流
        │
        └─→ 3. correct_detection（自动纠正）
              → detection-service
              → 误报修正：diaoche → dazhuangji（类别修正）
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

**关键配置项**：

```yaml
data:
  training_dir: "训练集路径"
  test_dir: "误报测试集路径"

llm:
  protocol: "openai"           # LLM 协议：openai / anthropic / ollama
  api_url: "https://..."       # API 地址
  api_key: "..."
  model: "mimo-v2.5"           # 模型名
  timeout: 300
  temperature: 0.1

detection:
  api_url: "http://192.168.99.180:49080/api/request/objectDetection"
  model_id: "wajueji_chanche_202606111433"

semantic:
  model_name: "BGE-VL-large 模型路径"
  device: "cuda"
  cache_dir: "./semantic_cache"
```

### 8.3 Embedding 缓存

BGE-VL-large 的图像 embedding 计算较慢（训练集 10104 张约需 20 分钟），采用**文件缓存**策略：
- 缓存 key：`MD5(维度名 + 排序后的图片路径列表)`
- 缓存格式：numpy `.npy` 文件 + `_paths.json` 元数据
- 缓存目录：`./semantic_cache/`（config.yaml 可配置）
- 同一批图片 + 同一维度不会重复计算
- 首次启动后缓存命中，初始化从 ~20 分钟降至 ~37 秒

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

不是简单地判断"这个样本值不值得回流"，而是从类别分布、光照、视角、模糊、天气、时间、环境 7 个维度综合分析，给出可解释的归因结果和回流建议。每个维度会标注是否是训练集的覆盖缺口，帮助定位数据缺陷。

### 6. Agent 编排 + 微服务执行的分层设计

Agent 层（GPT-5.5）负责理解用户意图和工具选择，服务层（MiMo v2.5）负责具体计算。两层使用不同的 LLM，各取所长：Agent 需要强推理能力，服务需要快速稳定的结构化输出。PI Agent 提供 9 个工具，通过 Skill 定义标准化工作流，Agent 按需加载 Skill 并调用对应工具。

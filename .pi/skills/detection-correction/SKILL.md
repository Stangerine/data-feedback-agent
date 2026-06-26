---
name: detection-correction
description: >
  目标检测结果纠正与可视化。当用户提到"纠正检测结果"、"修复误报"、"补漏报的bbox"、
  "生成纠正对比图"、"correct detection"、"大模型纠错"、"输出漏报目标框"、
  "小模型和大模型的对比图"时使用此技能。
  调用小模型API检测后，用大模型纠正误报类别、补充漏报目标的bbox，并生成可视化对比图。
  与 detection-review 的区别：review 只做校验分析，correction 会生成纠正后的标注图和JSON。
---

# 目标检测纠正

对小模型检测结果进行大模型纠正：修正误报类别、补充漏报目标的边界框，并保存小模型标注图与大模型纠正图的对比可视化。

## 工具

| 工具 | 用途 | 何时使用 |
|------|------|----------|
| `check_detection_service` | 检查服务状态 | 每次纠正前先调用 |
| `correct_detection` | 完整纠正 | 核心工具：检测 + LLM纠错 + 可视化 |
| `verify_detection` | 仅校验 | 只需要分析不需要生成图时 |

## 纠正流程

1. 调用 `check_detection_service` 确认服务在线
2. 调用 `correct_detection`，参数：
   - `image_path`: 图片的本地绝对路径
3. 解读纠正结果并向用户报告

### 服务不可用时

提示用户检查 detection-service（端口 8001）是否启动：
```bash
cd services/detection-service && bash start.sh
```

## 结果字段说明

### 纠正后的目标列表 (corrections)

每个目标包含 `source` 字段，标识来源：

| source | 含义 | 说明 |
|--------|------|------|
| `small_model` | 小模型原始结果 | 未被大模型修改，保留原样 |
| `llm_corrected` | 类别纠正 | 小模型框保留，但类别被大模型修正 |
| `llm_added` | 新增目标 | 小模型漏检，大模型补充的 bbox |

### 漏报 bbox 说明

大模型输出的漏报 bbox 是归一化坐标 (0-1)，经服务端转换为像素坐标。这些 bbox 用于辅助复核，精度可能不如小模型的检测框，需要人工确认。

### 可视化输出

服务会保存以下文件到 `correction_results/` 目录：

| 文件 | 内容 |
|------|------|
| 小模型标注图 | 原始 YOLO 检测结果的可视化 |
| 大模型纠正图 | 纠正后的结果（修正类别 + 新增目标） |
| 结果 JSON | 结构化的纠正数据 |

## 报告格式

```
纠正结果：[图片名]
─────────────────────
小模型目标数：N
纠正后目标数：M

误报修正：
  1. 第N条 [old_class] → [new_class]
     原因：[reason]

漏报新增：
  1. [class_name] bbox=[x1,y1,x2,y2]，置信度：[confidence]

可视化：
  小模型图：[path]
  纠正图：[path]
  JSON：[path]
```

## 与 detection-review 的关系

- **detection-review**: 只做校验分析，输出误报/漏报的结构化报告，不生成图片
- **detection-correction**: 在校验基础上，生成纠正后的标注图和可视化对比图

如果用户只需要"看看检测对不对"，用 detection-review。如果用户需要"修复并生成对比图"，用 detection-correction。

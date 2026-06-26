---
name: detection-review
description: >
  目标检测结果二次校验。当用户提到"校验检测结果"、"检查误报"、"审核检测"、"verify detection"、
  "看看检测对不对"、"检查这张图片的检测"、"误报漏报分析"时使用此技能。
  也适用于用户提供了图片路径并要求检查目标检测质量的场景。
  使用多模态大模型(MiMo v2.5)对小模型(YOLOv5)的检测结果进行二次校验，输出结构化的误报/漏报分析。
---

# 目标检测二次校验

使用大模型对小模型的检测结果进行二次校验，识别误报(false positive)和漏报(missed detection)。

## 工具

| 工具 | 用途 | 何时使用 |
|------|------|----------|
| `check_detection_service` | 检查服务状态 | 每次校验前先调用，确认服务可用 |
| `verify_detection` | 完整校验 | 有图片路径时使用，自动调检测API + LLM校验 |
| `verify_direct` | 直接校验 | 已有检测结果(JSON)时使用，跳过检测API |

## 校验流程

### 完整校验（推荐）

用户提供了图片路径：

1. 调用 `check_detection_service` 确认服务在线
2. 调用 `verify_detection`，参数：
   - `image_path`: 图片的本地绝对路径（如 `E:\zzq\误报\20250304175558461.jpg`）
3. 解读结果并向用户报告

### 直接校验

用户已提供检测结果 JSON，或需要跳过检测 API：

1. 调用 `verify_direct`，参数：
   - `image_path`: 图片路径
   - `detections`: 检测结果数组，每项包含 `class_name`、`confidence`、`bbox`（可选）

### 服务不可用时

如果 `check_detection_service` 返回错误：
- 提示用户检查 detection-service 是否启动（端口 8001）
- 启动命令：`cd services/detection-service && bash start.sh`

## 结果字段说明

### 误报 (false_positives)

小模型检测到了目标，但大模型判断为错误：

| 字段 | 含义 |
|------|------|
| `detection_index` | 对应原始检测结果的序号 |
| `reported_class` | 小模型给出的类别 |
| `actual_class` | 大模型判断的实际类别 |
| `reason` | 误报原因（中文） |
| `confidence` | 大模型的确信度 (0-1) |

### 漏报 (missed_detections)

小模型未检测到，但大模型发现存在目标：

| 字段 | 含义 |
|------|------|
| `actual_class` | 实际存在的目标类别 |
| `actual_class_cn` | 中文类别名 |
| `location` | 在图片中的位置描述 |
| `region_hint` | 归一化坐标范围 (x_min, y_min, x_max, y_max) |
| `confidence_level` | 大模型的确信度: high / medium / low |

### 总体评估 (overall_assessment)

| 字段 | 含义 |
|------|------|
| `detection_quality` | 检测质量: good / fair / poor |
| `false_positive_count` | 误报数量 |
| `missed_detection_count` | 漏报数量 |
| `summary` | 一句话总结 |

## 报告格式

向用户汇报时使用此结构：

```
校验结果：[图片名]
─────────────────────
检测质量：[good/fair/poor]
误报数量：X 个
漏报数量：X 个

误报详情：
  1. 第N条：[reported_class] → [actual_class]
     原因：[reason]

漏报详情：
  1. [actual_class_cn]：[location]
     确信度：[confidence_level]

总结：[summary]
```

## 9 类施工车辆参考

| ID | 英文名 | 中文名 | 视觉特征 |
|----|--------|--------|----------|
| 0 | wajueji | 挖掘机 | 黄色/红色大型履带式，有铲斗 |
| 1 | chanche | 铲车 | 带铲斗的轮式装载机 |
| 2 | dazhuangji | 打桩机 | 高耸柱状打桩设备，刚性桅杆 |
| 3 | yaluji | 压路机 | 大型钢轮碾压设备 |
| 4 | diaoche | 吊车 | 带吊臂的起重设备，有钢丝绳和吊钩 |
| 5 | gaokongche | 高空车 | 带升降平台的作业车 |
| 6 | youguanche | 油罐车 | 圆柱形罐体运输车 |
| 7 | yunshuche | 运输车 | 自卸卡车/渣土车 |
| 8 | other | 其他 | 不属于以上类别 |

这些信息有助于理解大模型的校验判断——例如打桩机和吊车外形相似，是常见的混淆对。

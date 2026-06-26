---
name: data-analysis
description: >
  训练数据质量分析与误报归因。当用户提到"数据分析"、"误报归因"、"回流分析"、"训练集分析"、
  "哪些数据值得回流"、"误报原因分析"、"data analysis"、"数据质量"、"训练数据分布"、
  "补充训练数据"、"数据回流建议"时使用此技能。
  也适用于用户问"为什么这个类别误报多"、"训练集缺什么数据"等关于数据质量的问题。
  通过 BGE-VL-large embedding + LLM 归因，从7个维度分析误报/漏报的根本原因，
  判断是数据缺陷还是其他因素，给出是否建议回流训练集的判断。
---

# 训练数据质量分析与误报归因

从类别分布、光照、视角、模糊、天气、时间、环境等 7 个维度分析误报/漏报的根本原因，
判断是训练数据缺陷还是其他因素导致，给出数据回流建议。

## 工具

| 工具 | 用途 | 参数 |
|------|------|------|
| `check_analysis_service` | 检查服务状态 | 无参数 |
| `analyze_dataset` | 数据集画像 | `data_dir`: 数据集目录路径 |
| `analyze_single` | 单图归因分析 | 见下方详细说明 |
| `analyze_batch` | 批量归因分析 | `training_dir`(可选), `test_dir`(可选) |
| `export_analysis` | 导出结果 | `format`: json/csv, `min_level`: feedback/no_feedback/all |

## 核心流程：单图归因分析

这是最常用的流程，结合 detection-service 的校验结果进行深度归因：

### 第一步：获取检测和校验结果

调用 `verify_detection`（来自 detection-review 技能）获取：
- `detections`: 小模型检测结果
- `verification.data.false_positives`: 误报列表
- `verification.data.missed_detections`: 漏报列表

### 第二步：调用归因分析

调用 `analyze_single`，参数：

```json
{
  "image_path": "图片绝对路径",
  "detections": [{"class_name": "diaoche", "confidence": 0.89}],
  "false_positives": [{"class_name": "diaoche", "confidence": 0.9, "reason": "..."}],
  "false_negatives": [{"class_name": "diazhuangji", "reason": "..."}],
  "verification_result": { ... }
}
```

- `image_path` (必填): 图片的本地绝对路径
- `detections` (可选): 当前检测结果
- `false_positives` (可选): 误报列表，每项需 `class_name` 和 `confidence`
- `false_negatives` (可选): 漏报列表，每项需 `class_name`
- `verification_result` (可选): detection-service 的完整校验结果 JSON

### 第三步：解读归因结果

归因结果包含：

| 字段 | 含义 |
|------|------|
| `attribution_type` | 归因类型（见下方表格） |
| `confidence` | 归因置信度 (0-1) |
| `main_cause_dimension` | 主要原因维度 |
| `should_feedback` | 是否建议回流训练集 |
| `feedback_suggestion` | 回流建议（中文） |
| `dimension_attributions` | 各维度的详细归因 |

## 批量分析流程

对整个测试集进行批量归因：

1. 调用 `check_analysis_service` 确认服务已初始化（首次需要较长时间计算训练集 embedding）
2. 调用 `analyze_batch`，可选参数：
   - `training_dir`: 训练集目录（默认使用 config.yaml 中的配置）
   - `test_dir`: 测试集目录（默认使用 config.yaml 中的配置）
3. 调用 `export_analysis` 导出结果

## 归因类型

| 归因类型 | 含义 | 回流建议 |
|----------|------|----------|
| 数据缺陷 | 训练集缺少该类别/场景的样本 | **强烈建议回流** |
| 背景干扰 | 背景物体与目标视觉特征相似 | 建议回流 |
| 类间混淆 | 不同目标类别的外观相似 | 建议回流 |
| 遮挡截断 | 目标被遮挡或画面截断 | 视严重程度而定 |
| 环境因素 | 特殊环境（雨天、夜晚等）导致 | 视频率而定 |
| 标注错误 | 原始标注本身有误 | 需人工确认 |

## 7 个分析维度

每个维度会给出该图片的分类，以及训练集中该分类的覆盖比例。如果覆盖比例低且是误报主因，则标记为"覆盖缺口"。

| 维度 | 分类值 | 说明 |
|------|--------|------|
| class | 对应9类车辆 | 训练集中该类别的样本占比 |
| lighting | dim / bright / moderate | 光照条件 |
| viewpoint | rear / front / overhead / side | 拍摄视角 |
| blur | motion-blur / out-of-focus / sharp | 模糊程度 |
| weather | rain / snow / cloudy / clear / fog | 天气条件 |
| timeOfDay | day / night / dusk | 拍摄时间段 |
| environment | construction-site / urban-street / indoor / aerial-scene / rural-field | 环境类型 |

## 结果报告结构

向用户汇报归因结果时：

1. **一句话结论**: 归因类型 + 是否建议回流
2. **主因维度**: 哪个维度是主要贡献因素，训练集覆盖情况
3. **各维度详情**: 列出每个维度的分类和覆盖情况，标记覆盖缺口
4. **回流建议**: 具体建议补充什么数据

## 9 类施工车辆

归因分析会引用这些类别名称：

| ID | 英文名 | 中文名 | 视觉特征 |
|----|--------|--------|----------|
| 0 | wajueji | 挖掘机 | 黄色/红色大型履带式，有铲斗 |
| 1 | chanche | 铲车 | 带铲斗的轮式装载机 |
| 2 | dazhuangji | 打桩机 | 高耸柱状打桩设备，刚性桅杆 |
| 3 | yaluji | 压路机 | 大型钢轮碾压设备 |
| 4 | diaoche | 吊车 | 带吊臂的起重设备 |
| 5 | gaokongche | 高空车 | 带升降平台的作业车 |
| 6 | youguanche | 油罐车 | 圆柱形罐体运输车 |
| 7 | yunshuche | 运输车 | 自卸卡车/渣土车 |
| 8 | other | 其他 | 不属于以上类别 |

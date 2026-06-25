---
name: detection-correction
description: 当用户要求纠正目标检测结果、修复误报漏报、给漏报目标补bbox、生成小模型与大模型纠正后的可视化对比图时使用此技能。优先调用 correct_detection 工具完成真实小模型API检测、大模型纠错和图片保存。
---

# 目标检测纠正技能

用于把“小模型检测结果”经过多模态大模型二次纠正后变成可复核的结果。

## 可用工具

- `check_detection_service` — 检查检测服务状态
- `correct_detection` — 调小模型 API，使用大模型纠正误报类别、补充漏报 bbox，并保存对比图
- `verify_detection` — 只做误报/漏报审核，不生成纠正图

## 标准流程

当用户要求“纠正检测结果”“修复误报漏报”“输出漏报 bbox”“生成对比图”：

1. 先调用 `check_detection_service` 确认服务可用。
2. 调用 `correct_detection`，传入图片绝对路径。
3. 向用户报告：
   - 小模型检测数量
   - 纠正后目标数量
   - 误报类别修正摘要
   - 漏报新增目标摘要
   - 小模型标注图路径
   - 大模型纠正图路径
   - 结果 JSON 路径

## 解读规则

- `source=small_model`：小模型原始结果，未被大模型修改。
- `source=llm_corrected`：小模型框被保留，但类别被大模型纠正。
- `source=llm_added`：小模型漏检，大模型新增 bbox。
- 漏报 bbox 是大模型输出的归一化坐标经服务端转换后的像素坐标，只用于辅助复核。

## 报告格式

```text
纠正结果：[图片名]
─────────────────────
小模型目标数：N
纠正后目标数：M

误报修正：
  1. 第N条 old_class → new_class，原因：...

漏报新增：
  1. class_name bbox=[x1,y1,x2,y2]，置信度：...

可视化：
  小模型图：...
  纠正图：...
  JSON：...
```

# 使用 OpenAI 多模态大模型 API 做目标检测并输出 BBox 的方案调研

> 版本：v1.0  
> 适用场景：技术预研、方案评审、Demo 实现、开放词汇目标定位、语义辅助检测  
> 核心结论：可以使用 OpenAI 多模态大模型 API 实现“图片输入 → 目标识别 → BBox JSON 输出”，但更适合作为开放词汇/语义定位能力，而不是传统高精度目标检测器的完全替代。

---

## 1. 背景与目标

目标是调研并设计一种使用 OpenAI 多模态大模型 API 进行目标检测的方法，使模型能够：

1. 接收图片输入；
2. 根据指定类别或自然语言目标描述识别图中目标；
3. 输出结构化的 bounding box；
4. 便于业务系统后续做可视化、裁剪、复核、告警或人工标注辅助。

典型输出形式如下：

```json
{
  "image_width": 1280,
  "image_height": 720,
  "objects": [
    {
      "label": "person",
      "bbox": {
        "x_min": 0.123,
        "y_min": 0.214,
        "x_max": 0.301,
        "y_max": 0.824
      },
      "confidence": 0.82,
      "description": "standing person on the left"
    }
  ]
}
```

---

## 2. 总体结论

OpenAI 多模态模型可以通过以下组合实现 bbox 输出：

```text
图片输入
  ↓
多模态大模型理解图片
  ↓
Prompt 明确目标类别、坐标系和输出规则
  ↓
Structured Outputs / JSON Schema 约束返回格式
  ↓
服务端校验、坐标转换、过滤、NMS、评估
```

推荐技术路线：

| 模块 | 推荐方案 |
|---|---|
| 图片输入 | Responses API 的 `input_image` |
| 图像细节 | 空间定位任务优先使用 `detail: "original"` |
| 输出约束 | Structured Outputs / JSON Schema |
| 坐标格式 | 归一化 `xyxy` 坐标 |
| 后处理 | clamp、格式校验、置信度过滤、NMS、IoU 评估 |
| 适用定位 | 开放词汇检测、语义目标定位、辅助标注、低频类别定位 |
| 不建议单独承担 | 高精度实时检测、密集小目标检测、高风险自动决策 |

---

## 3. 官方能力基础

### 3.1 图片输入

OpenAI Responses API 支持向模型传入图片输入，图片内容可以通过：

- 图片 URL；
- base64 data URL；
- 上传后的 file id。

核心输入结构：

```json
{
  "type": "input_image",
  "image_url": "data:image/jpeg;base64,...",
  "detail": "original"
}
```

相关官方文档：

- Images and vision guide: https://developers.openai.com/api/docs/guides/images-vision
- Responses API reference: https://developers.openai.com/api/reference/resources/responses/methods/create

### 3.2 图像细节等级 detail

`detail` 参数用于控制模型处理图片的细节级别，可选值通常包括：

```text
low
high
auto
original
```

对于目标定位、bbox、点击精度、空间敏感任务，建议优先使用：

```json
"detail": "original"
```

原因：

- bbox 依赖空间定位；
- `low` 可能丢失小目标和边缘细节；
- `original` 更适合大图、密集图、空间敏感图像。

### 3.3 结构化输出

为了稳定拿到 bbox JSON，不建议只靠普通 prompt 让模型“返回 JSON”。

更推荐使用：

```text
Structured Outputs / JSON Schema
```

这样可以强制输出字段结构，例如：

- `objects` 必须是数组；
- `bbox` 必须包含 `x_min/y_min/x_max/y_max`；
- 坐标必须是数字；
- `confidence` 必须是数字；
- 不允许多余字段。

相关官方文档：

- Structured Outputs guide: https://developers.openai.com/api/docs/guides/structured-outputs

---

## 4. BBox 坐标设计

### 4.1 推荐格式：归一化 xyxy

推荐使用归一化坐标，而不是让模型直接输出原图像素坐标。

```text
bbox = {
  x_min: 左边界,
  y_min: 上边界,
  x_max: 右边界,
  y_max: 下边界
}
```

坐标约定：

```text
坐标范围：[0, 1]
原点：左上角
x 轴：向右
y 轴：向下
bbox 覆盖目标的可见区域
格式：xyxy
```

示例：

```json
{
  "bbox": {
    "x_min": 0.12,
    "y_min": 0.20,
    "x_max": 0.38,
    "y_max": 0.85
  }
}
```

### 4.2 为什么优先用归一化坐标

原因包括：

1. 大模型内部可能对图片进行缩放或 token 化处理；
2. 直接输出像素坐标更容易受到缩放影响；
3. 归一化坐标便于跨分辨率图片使用；
4. 后端可以统一转换成像素坐标；
5. 便于可视化和评估。

### 4.3 坐标转换

后端将归一化坐标转换为像素坐标：

```python
px_x_min = round(x_min * image_width)
px_y_min = round(y_min * image_height)
px_x_max = round(x_max * image_width)
px_y_max = round(y_max * image_height)
```

---

## 5. 推荐输出 Schema

### 5.1 JSON Schema 结构

```json
{
  "image_width": 1280,
  "image_height": 720,
  "objects": [
    {
      "label": "person",
      "bbox": {
        "x_min": 0.123,
        "y_min": 0.214,
        "x_max": 0.301,
        "y_max": 0.824
      },
      "confidence": 0.82,
      "description": "standing person on the left"
    }
  ]
}
```

### 5.2 字段说明

| 字段 | 类型 | 说明 |
|---|---:|---|
| `image_width` | int | 原图宽度 |
| `image_height` | int | 原图高度 |
| `objects` | array | 检测到的目标列表 |
| `label` | string | 目标类别 |
| `bbox.x_min` | float | 左边界，归一化 |
| `bbox.y_min` | float | 上边界，归一化 |
| `bbox.x_max` | float | 右边界，归一化 |
| `bbox.y_max` | float | 下边界，归一化 |
| `confidence` | float | 模型自评置信度，0 到 1 |
| `description` | string | 目标的简短描述，用于调试或人工复核 |

---

## 6. Prompt 设计

### 6.1 固定类别检测 Prompt 模板

```text
You are an object detection system.

Detect only these target classes:
{target_classes}

Return bounding boxes as normalized coordinates in [0, 1].

Coordinate system:
- origin is top-left
- x increases to the right
- y increases downward
- bbox format is x_min, y_min, x_max, y_max
- bbox should tightly enclose the visible part of the object
- do not infer hidden or occluded full object extent
- if no target object is visible, return objects: []

Rules:
- Do not include objects outside the target classes.
- If uncertain, lower the confidence score.
- If an object is partially occluded, box only the visible region.
- Avoid duplicate boxes for the same object.
```

### 6.2 开放词汇检测 Prompt 模板

适用于“找出图中所有异常部件”“找出可能有安全风险的物体”等任务。

```text
You are a visual inspection and open-vocabulary detection system.

Task:
Find all objects or regions that match the following description:
{target_description}

Return normalized bounding boxes in [0, 1].

For each detection, include:
- label
- bbox
- confidence
- description
- reason why this region matches the target description

Only return visible regions.
If no matching object is visible, return an empty objects array.
```

### 6.3 Prompt 注意事项

建议明确以下约束：

```text
1. 坐标系
2. bbox 格式
3. 是否框住可见区域还是完整推测区域
4. 是否允许开放类别
5. 无目标时返回空数组
6. 是否允许重复框
7. 对遮挡、小目标、模糊目标的处理规则
```

---

## 7. API 实现示例

下面示例展示如何用 Python SDK、图片输入和 Pydantic 结构化输出完成 bbox 检测。

> 注意：模型名、SDK 版本、字段写法需要以实际项目中的 OpenAI SDK 版本和官方文档为准。

```python
import base64
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field
from openai import OpenAI

client = OpenAI()


class BBox(BaseModel):
    x_min: float = Field(description="Normalized left coordinate, 0 to 1")
    y_min: float = Field(description="Normalized top coordinate, 0 to 1")
    x_max: float = Field(description="Normalized right coordinate, 0 to 1")
    y_max: float = Field(description="Normalized bottom coordinate, 0 to 1")


class DetectedObject(BaseModel):
    label: str
    bbox: BBox
    confidence: float = Field(description="Confidence score from 0 to 1")
    description: str


class DetectionResult(BaseModel):
    image_width: int
    image_height: int
    objects: List[DetectedObject]


def image_to_data_url(path: str) -> str:
    image_bytes = Path(path).read_bytes()
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    suffix = Path(path).suffix.lower()
    if suffix == ".png":
        mime = "image/png"
    else:
        mime = "image/jpeg"

    return f"data:{mime};base64,{b64}"


def detect_objects(
    image_path: str,
    image_width: int,
    image_height: int,
    target_classes: list[str],
) -> DetectionResult:
    target_text = ", ".join(target_classes)

    prompt = f"""
You are an object detection system.

Detect only these target classes:
{target_text}

Return bounding boxes as normalized coordinates in [0, 1].

Coordinate system:
- origin is top-left
- x increases to the right
- y increases downward
- bbox format is x_min, y_min, x_max, y_max
- bbox should tightly enclose the visible part of the object
- if no target object is visible, return objects: []

Original image size:
width={image_width}, height={image_height}

Do not include objects outside the target classes.
Avoid duplicate boxes for the same object.
"""

    response = client.responses.parse(
        model="gpt-5.5",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": image_to_data_url(image_path),
                        "detail": "original",
                    },
                ],
            }
        ],
        text_format=DetectionResult,
    )

    return response.output_parsed


if __name__ == "__main__":
    result = detect_objects(
        image_path="test.jpg",
        image_width=1280,
        image_height=720,
        target_classes=["person", "car", "traffic light"],
    )

    print(result.model_dump_json(indent=2))
```

---

## 8. 后处理设计

模型输出 bbox 后，服务端必须做校验和清洗。

### 8.1 坐标合法性校验

```python
def validate_and_convert(obj, width: int, height: int):
    box = obj.bbox

    x1 = max(0.0, min(1.0, box.x_min))
    y1 = max(0.0, min(1.0, box.y_min))
    x2 = max(0.0, min(1.0, box.x_max))
    y2 = max(0.0, min(1.0, box.y_max))

    if x2 <= x1 or y2 <= y1:
        return None

    return {
        "label": obj.label,
        "confidence": obj.confidence,
        "bbox_xyxy_pixels": [
            round(x1 * width),
            round(y1 * height),
            round(x2 * width),
            round(y2 * height),
        ],
        "description": obj.description,
    }
```

### 8.2 推荐后处理步骤

```text
1. 坐标 clamp 到 [0, 1]
2. 删除 x_max <= x_min 或 y_max <= y_min 的非法框
3. 按 confidence 做阈值过滤
4. 按 label 白名单过滤
5. 对同类目标做 NMS 或去重
6. 过滤面积过小或异常大的框
7. 转换成像素坐标
8. 绘制检测结果用于可视化复核
```

### 8.3 置信度策略

大模型返回的 `confidence` 是模型自评置信度，不等价于传统检测模型的 calibrated confidence。

建议：

```text
Demo 阶段：confidence >= 0.5
人工复核阶段：confidence >= 0.4
自动化业务流程：confidence >= 0.7，并结合其他检测器或规则
高风险场景：不能只依赖该置信度
```

---

## 9. 评估方法

### 9.1 离线评估集

建议准备一批人工标注的图片作为验证集：

```text
数量：至少 100~500 张起步
覆盖：不同光照、角度、遮挡、尺度、密度、背景
标注：目标类别 + bbox
指标：IoU、Precision、Recall、F1、漏检率、误检率
```

### 9.2 IoU 评估

IoU 用于衡量预测框和人工标注框的重叠程度：

```text
IoU = 预测框与标注框交集面积 / 预测框与标注框并集面积
```

常见阈值：

```text
IoU >= 0.5：粗定位可接受
IoU >= 0.75：定位较好
IoU >= 0.9：像素级较严格
```

### 9.3 重点观察指标

除了传统指标，还建议观察：

```text
1. 小目标漏检率
2. 遮挡目标表现
3. 密集目标重复框比例
4. 同一 prompt 多次调用的一致性
5. 类别混淆情况
6. 开放词汇目标的语义匹配质量
7. 输出 JSON 格式稳定性
```

---

## 10. 适合场景

### 10.1 开放词汇目标检测

例如：

```text
找出图片中所有“可能损坏的零件”
找出“穿红色衣服的人”
找出“桌面上不该出现的物品”
找出“有安全隐患的位置”
```

这类任务传统检测器往往需要训练特定类别，而多模态大模型可以直接用自然语言描述目标。

### 10.2 辅助标注

可用于：

```text
自动生成初始 bbox
人工快速修正
降低标注成本
加快数据集构建
```

### 10.3 语义复核

传统检测模型负责初筛，大模型负责判断：

```text
这个目标是否真的属于指定语义？
这个区域是否异常？
这个检测结果是否合理？
是否需要人工介入？
```

### 10.4 UI / 文档 / 截图定位

可用于：

```text
定位按钮
定位表格
定位图表
定位页面元素
定位发票或表单中的关键区域
```

---

## 11. 不适合单独承担的场景

不建议仅依赖多模态大模型做以下任务：

```text
1. 自动驾驶感知
2. 医疗影像诊断
3. 安防实时告警
4. 工业高精度缺陷检测
5. 密集小目标检测
6. 高 FPS 实时视频检测
7. 需要严格 mAP 指标的生产检测系统
8. 高风险自动决策系统
```

原因：

```text
1. 空间定位精度不如专用检测模型稳定
2. 多次调用可能存在不一致性
3. 成本和延迟通常高于本地检测模型
4. 小目标、密集目标、遮挡目标容易出现漏检或框偏移
5. 模型 confidence 不等价于传统检测器置信度
```

---

## 12. 推荐系统架构

### 12.1 原型架构

```text
前端/服务端上传图片
  ↓
OpenAI 多模态模型
  ↓
Structured JSON bbox
  ↓
后处理
  ↓
前端绘制 bbox
```

适合：

```text
Demo
PoC
内部工具
低频调用
开放词汇测试
```

### 12.2 生产增强架构

```text
图片输入
  ↓
传统检测器生成候选框
  ↓
OpenAI 多模态模型做语义判断/复核/补充
  ↓
规则引擎 + 后处理
  ↓
人工审核或业务动作
```

推荐组合：

| 模块 | 职责 |
|---|---|
| YOLO / DETR / Grounding DINO | 高效候选检测 |
| OpenAI 多模态模型 | 开放语义理解、复杂描述判断、异常解释 |
| 后处理模块 | 坐标校验、NMS、阈值过滤 |
| 评估模块 | IoU、Precision、Recall、稳定性评估 |
| 人审模块 | 低置信度或高风险样本复核 |

### 12.3 混合检测策略

```text
1. 常见类别：传统检测器主导
2. 长尾类别：OpenAI 模型补充
3. 语义复杂类别：OpenAI 模型判断
4. 高风险结果：人工复核
5. 低置信度结果：不自动触发业务动作
```

---

## 13. 成本与性能考虑

需要关注：

```text
1. 图片大小
2. detail 等级
3. 模型选择
4. 请求并发
5. 批处理策略
6. 是否需要缓存结果
7. 是否可以先用传统模型裁剪 ROI 再调用大模型
```

优化思路：

```text
1. 对大图先压缩到业务可接受尺寸
2. 对空间敏感任务使用 original
3. 对粗分类任务使用 high 或 auto
4. 先用本地模型或规则过滤无关图片
5. 对候选区域 crop 后再调用模型
6. 对重复图片做 hash 缓存
7. 对低价值场景降低调用频率
```

---

## 14. 风险与限制

### 14.1 空间定位限制

多模态大模型并不是专用目标检测器，在精确空间定位任务中可能存在：

```text
1. bbox 偏移
2. 框过大或过小
3. 小目标漏检
4. 多目标重复框
5. 遮挡目标判断不稳定
6. 密集目标计数不准
```

### 14.2 输出稳定性

即使使用结构化输出，也只能约束 JSON 格式，不能保证检测结果绝对正确。

建议：

```text
1. 所有 bbox 必须后处理
2. 不要直接信任坐标
3. 不要直接信任 confidence
4. 对关键场景保留人工复核
5. 建立离线评估集
```

### 14.3 合规与安全

高风险场景需谨慎，例如：

```text
1. 医疗诊断
2. 安防执法
3. 自动驾驶
4. 人身安全相关判断
5. 金融、保险、法律自动决策
```

---

## 15. 实施路线图

### 阶段 1：PoC 验证

目标：

```text
证明 OpenAI 多模态模型可以输出可用 bbox
```

任务：

```text
1. 准备 20~50 张测试图片
2. 设计固定类别 prompt
3. 实现 Structured Outputs
4. 实现 bbox 绘制
5. 人工观察定位效果
```

交付物：

```text
1. API Demo
2. bbox 可视化页面
3. 初步问题列表
```

### 阶段 2：小规模评估

目标：

```text
判断是否满足业务精度要求
```

任务：

```text
1. 准备 100~500 张标注数据
2. 计算 IoU、Precision、Recall
3. 调整 prompt
4. 测试不同 detail 配置
5. 测试不同目标类别
```

交付物：

```text
1. 评估报告
2. 最佳 prompt
3. 推荐阈值
4. 错误案例分析
```

### 阶段 3：生产化设计

目标：

```text
将能力接入业务系统
```

任务：

```text
1. 加入后处理模块
2. 加入缓存和重试
3. 加入人工复核流
4. 与传统检测器组合
5. 监控成本、延迟和错误率
```

交付物：

```text
1. 服务接口
2. 监控指标
3. 回归测试集
4. 人审流程
5. 上线方案
```

---

## 16. 推荐接口设计

### 16.1 请求

```json
{
  "image": "base64 or image_url",
  "image_width": 1280,
  "image_height": 720,
  "target_classes": ["person", "car", "traffic light"],
  "mode": "fixed_class",
  "detail": "original"
}
```

### 16.2 响应

```json
{
  "status": "success",
  "image_width": 1280,
  "image_height": 720,
  "objects": [
    {
      "label": "person",
      "confidence": 0.82,
      "bbox_normalized": [0.123, 0.214, 0.301, 0.824],
      "bbox_pixels": [157, 154, 385, 593],
      "description": "standing person on the left"
    }
  ],
  "warnings": []
}
```

### 16.3 错误响应

```json
{
  "status": "error",
  "error_code": "INVALID_BBOX_OUTPUT",
  "message": "Model returned invalid bbox coordinates.",
  "raw_output": {}
}
```

---

## 17. 最终建议

如果业务目标是快速实现开放词汇目标定位，可以采用：

```text
OpenAI Responses API
+ input_image
+ detail: original
+ Structured Outputs
+ normalized xyxy bbox
+ 后处理与评估
```

如果业务目标是高精度、低延迟、可规模化的目标检测，推荐采用混合架构：

```text
传统检测模型负责高效检测
OpenAI 多模态模型负责语义理解、开放类别判断、异常解释和复核
```

最终定位：

```text
OpenAI 多模态模型适合做“语义检测/辅助检测组件”
不建议在高精度目标检测场景中完全替代专用检测模型
```

---

## 18. 参考资料

1. OpenAI Images and Vision Guide  
   https://developers.openai.com/api/docs/guides/images-vision

2. OpenAI Structured Outputs Guide  
   https://developers.openai.com/api/docs/guides/structured-outputs

3. OpenAI Responses API Reference  
   https://developers.openai.com/api/reference/resources/responses/methods/create

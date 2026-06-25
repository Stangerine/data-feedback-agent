# test_services.py 测试逻辑总结

## 概述

`test_services.py` 是一个端到端集成测试，使用真实图片按顺序跑通 detection-service 和 data-analysis-service 的完整业务逻辑，验证"检测 → 校验 → 纠正 → 归因"全链路是否正常。

## 整体流程

```
main()
  ├── test_detection_service()        ← 阶段一：检测校验
  │     ├── 步骤1: 小模型检测 (YOLO API)
  │     ├── 步骤2: 大模型校验 (LLM)
  │     └── 步骤3: 自动纠正
  │     └── 输出 → tests/detection_results.json
  │
  └── test_data_analysis_service()    ← 阶段二：数据分析（依赖阶段一输出）
        ├── 初始化: 预计算训练集分布
        ├── 逐图归因分析
        └── 统计训练集分布
        └── 输出 → tests/attribution_results.json
```

## 阶段一：detection-service 测试

**函数**: `test_detection_service()` (第 20-149 行)

**输入**: 配置文件中的测试目录，取前 3 张图片

**三个步骤**:

| 步骤 | 组件 | 调用方法 | 输出 |
|------|------|----------|------|
| 步骤1 小模型检测 | `DetectionClient` | `detect(img_path, threshold)` | 检测框列表 (class_name, confidence) |
| 步骤2 大模型校验 | `Verifier` | `verify(img_path, detections)` | 误报列表、漏报列表、质量评估 |
| 步骤3 自动纠正 | `CorrectionService` | `correct(img_path, detections, verification)` | 纠正产物 (标注图等) |

**输出文件**: `tests/detection_results.json`，结构为：
```json
[
  {
    "image": "xxx.jpg",
    "detections": [...],
    "verification": { "success": true, "data": {...} },
    "correction": { "artifacts": {...} }
  }
]
```

## 阶段二：data-analysis-service 测试

**函数**: `test_data_analysis_service()` (第 152-295 行)

**输入**:
- 配置文件中的测试目录（同阶段一）
- 阶段一输出的 `detection_results.json`

**三个步骤**:

| 步骤 | 组件 | 说明 |
|------|------|------|
| 初始化 | `AnalysisPipeline.initialize()` | 预计算训练集类别分布和语义分布 |
| 逐图归因 | `pipeline.analyze_single()` | 输入检测+误报/漏报信息，输出归因结果 |
| 统计 | `pipeline.train_class_analysis` | 打印训练集类别和语义维度分布 |

**`analyze_single()` 输出字段**:

| 字段 | 含义 |
|------|------|
| `attribution_type` | 归因类型 |
| `confidence` | 置信度 |
| `main_cause_dimension` | 主因维度 |
| `should_feedback` | 是否建议回流训练集 |
| `feedback_suggestion` | 回流建议 |
| `dimension_attributions` | 各维度归因详情（类别、训练集占比、覆盖缺口） |

**输出文件**: `tests/attribution_results.json`

## 设计特点

1. **子进程隔离**: 两个测试都通过 `subprocess.run()` 在独立 Python 进程中运行，避免模块路径和依赖冲突
2. **动态脚本生成**: 用字符串模板 + `.format()` 生成完整测试脚本，写入临时文件 `_test_det.py` / `_test_analysis.py` 再执行
3. **数据串联**: detection 测试输出的 JSON 作为 analysis 测试的输入，模拟真实两阶段流水线
4. **容错处理**: 每个步骤都有 try/except，单张图片失败不阻塞后续处理，只记录错误继续执行
5. **计时统计**: 每个步骤记录耗时，便于性能分析

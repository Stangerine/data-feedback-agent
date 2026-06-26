# 服务逻辑测试报告

**测试时间**: 2026-06-26
**测试环境**: Windows Server 2025, CUDA GPU, conda `zzq` 环境
**测试图片**: `E:\zzq\误报\20250304175558461.jpg`

---

## 1. 测试概述

本次测试按照 `detection-service` 和 `data-analysis-service` 两个服务的完整业务逻辑进行端到端验证：

1. **detection-service**: 小模型检测 → 大模型校验 → 自动纠正
2. **data-analysis-service**: 训练集分布预计算 → 单图归因分析

---

## 2. 配置信息

| 配置项 | 值 |
|--------|-----|
| 检测 API | `http://192.168.99.180:49080/api/request/objectDetection` |
| 检测模型 ID | `wajueji_chanche_202606111433` |
| LLM 模型 | `mimo-v2.5` (小米 MiMo) |
| LLM API | `https://token-plan-cn.xiaomimimo.com/v1` |
| BGE 模型 | `E:\zzq\model\BGE-VL-large` |
| 训练集 | `E:\zzq\训练集\vehicle-13631-v18-cls9_split_80_10_10\train` (10104 张) |
| 测试集 | `E:\zzq\误报` (59 张) |

---

## 3. detection-service 测试结果

### 3.1 测试流程

| 步骤 | 操作 | 状态 | 耗时 |
|------|------|------|------|
| 步骤1 | 小模型检测 (YOLOv5 API) | OK | 0.86s |
| 步骤2 | 大模型校验 (MiMo v2.5) | OK | 18.92s |
| 步骤3 | 自动纠正 (MiMo v2.5) | OK | 15.98s |

### 3.2 小模型检测结果

检测到 **1 个目标**：

| # | 类别 | 置信度 | 边界框 (x1, y1, x2, y2) |
|---|------|--------|-------------------------|
| 1 | diaoche (吊车) | 0.89 | (437.2, 510.7, 616.9, 686.1) |

### 3.3 大模型校验结果

**误报 (False Positives): 1 个**

| 检测类别 | 实际类别 | 置信度 | 原因 |
|----------|----------|--------|------|
| diaoche (吊车) | dazhuangji (打桩机) | 0.90 | 该检测框覆盖的黄色设备具有高耸的刚性竖直桅杆结构，无柔性钢丝绳和吊钩，符合打桩机特征，而非吊车 |

**漏报 (Missed Detections): 1 个**

| 实际类别 | 置信度 | 位置 | 描述 |
|----------|--------|------|------|
| diaoche (吊车) | 0.95 | 画面左侧河道边 | 红色吊车，具有明显的起重臂和吊钩结构，主体清晰可见 |

**总体评估**: 检测质量 = **fair** (一般)

### 3.4 自动纠正结果

输出目录: `correction_results/20260626_150918_20250304175558461`

生成了小模型标注图和大模型纠正图的对比可视化。

---

## 4. data-analysis-service 测试结果

### 4.1 初始化

| 项目 | 耗时 | 说明 |
|------|------|------|
| 训练集分布预计算 | 37.72s | 使用 BGE-VL-large embedding 缓存 |

> 注: 首次运行（无缓存）约需 1284s (~21分钟)，有缓存后仅需 37.72s。

### 4.2 归因分析结果

| 项目 | 结果 |
|------|------|
| 归因类型 | **数据缺陷** |
| 置信度 | **0.90** |
| 主因维度 | **class (类别)** |
| 是否建议回流 | **是** |

**回流建议**:

> 应优先补充"打桩机"(dazhuangji)相关训练数据。理由：
> 1. 与当前图片场景相似（背景、颜色、挖掘视角）
> 2. 打桩机视角、场景下的打桩机样本极少，模型缺乏泛化能力
> 3. 正确标注并加入"打桩机"类别的关键样本

### 4.3 各维度归因详情

| 维度 | 分类 | 训练集占比 | 覆盖缺口 | 说明 |
|------|------|-----------|---------|------|
| **class** | dazhuangji | **1.1%** | **是** | 打桩机样本严重不足 |
| viewpoint | overhead | 31.2% | 否 | 俯视角度覆盖充足 |
| lighting | dim | 89.8% | 否 | 暗光场景为主 |
| blur | motion-blur | 86.1% | 否 | 运动模糊为主 |
| weather | rain | 52.4% | 否 | 雨天场景覆盖充足 |
| timeOfDay | dusk | 15.9% | 否 | 黄昏场景覆盖一般 |
| environment | construction-site | 96.0% | 否 | 工地场景覆盖充足 |

---

## 5. 训练集分布统计

### 5.1 类别分布

> 注: 本次测试未输出类别分布详情（需批量分析），以下为语义维度分布。

### 5.2 语义维度分布

| 维度 | 类别 | 占比 |
|------|------|------|
| **lighting** | dim (暗光) | 89.8% |
| | bright (明亮) | 5.4% |
| | moderate (适中) | 4.7% |
| **viewpoint** | rear (后方) | 52.3% |
| | overhead (俯视) | 31.2% |
| | front (前方) | 16.5% |
| **blur** | motion-blur (运动模糊) | 86.1% |
| | out-of-focus (失焦) | 9.9% |
| | sharp (清晰) | 4.0% |
| **weather** | rain (雨天) | 52.4% |
| | cloudy (多云) | 21.0% |
| | clear (晴天) | 17.0% |
| | snow (雪天) | 6.8% |
| | fog (雾天) | 2.8% |
| **timeOfDay** | day (白天) | 65.5% |
| | night (夜晚) | 18.6% |
| | dusk (黄昏) | 15.9% |
| **environment** | construction-site (工地) | 96.0% |
| | urban-street (城市街道) | 2.9% |
| | indoor (室内) | 0.5% |
| | aerial-scene (航拍) | 0.5% |
| | rural-field (乡村) | 0.1% |

---

## 6. 问题修复记录

测试过程中发现并修复了以下问题：

### 6.1 config.yaml 缺少 model_id

**问题**: 检测 API 返回 `Cannot invoke "CycModelRepository.getId()" because "modelRepository1" is null`
**原因**: `config.yaml` 中 `detection` 配置缺少 `model_id` 字段
**修复**: 添加 `model_id: "wajueji_chanche_202606111433"`

### 6.2 LLMConfig 缺少 protocol 字段

**问题**: `'LLMConfig' object has no attribute 'protocol'`
**原因**: `verifier.py` 和 `correction_service.py` 访问 `cfg.llm.protocol`，但 `LLMConfig` dataclass 未定义该字段
**修复**:
- `config.py`: `LLMConfig` 添加 `protocol: str = "openai"` 字段
- `verifier.py`: `_build_audit` 方法直接使用 `cfg.llm.protocol` 和 `cfg.llm.model`
- `correction_service.py`: `__init__` 方法直接使用 `cfg.llm.*` 扁平字段

### 6.3 data-analysis-service 缺少 openai 模块

**问题**: LLM 归因分析失败 `No module named 'openai'`
**原因**: conda 环境未安装 `openai` Python 包
**修复**: `pip install openai`

---

## 7. 结论

### 7.1 服务状态

| 服务 | 状态 | 说明 |
|------|------|------|
| detection-service | **正常** | 检测、校验、纠正全流程通过 |
| data-analysis-service | **正常** | 分布预计算、归因分析全流程通过 |

### 7.2 核心发现

本次测试的误报图片（20250304175558461.jpg）被系统正确识别为**数据缺陷**问题：

- **根因**: 打桩机(dazhuangji)在训练集中仅占 **1.1%**，样本严重不足
- **表现**: 小模型将打桩机误识别为吊车(diaoche)
- **建议**: 优先补充打桩机类别的训练数据

### 7.3 性能指标

| 指标 | 值 |
|------|-----|
| 小模型检测耗时 | 0.86s |
| 大模型校验耗时 | 18.92s |
| 大模型纠正耗时 | 15.98s |
| 训练集初始化 (有缓存) | 37.72s |
| 单图归因分析 | 34.04s |

---

## 8. 测试文件清单

| 文件 | 说明 |
|------|------|
| `tests/test_services.py` | 主测试脚本 |
| `tests/detection_results.json` | detection-service 测试结果 |
| `tests/attribution_results.json` | data-analysis-service 测试结果 |
| `tests/semantic_cache/` | BGE-VL-large embedding 缓存 |
| `config.yaml` | 全局配置文件 (已修复) |

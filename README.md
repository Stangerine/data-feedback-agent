# data-feedback-agent

Construction vehicle detection data feedback system — PI Agent + Python FastAPI 两层架构，用于检测结果校验和数据集质量分析。

## 功能

- **检测校验**：调用远程 YOLO API 检测，LLM 多模态分析误报和漏检
- **检测纠偏**：LLM 补充漏检的精确边界框，生成对比可视化图
- **数据集分析**：7 维度语义分析（光照、视角、清晰度、天气、时间、环境、类别覆盖）+ LLM 归因
- **交互式 Agent**：PI Agent 提供自然语言交互，自动调用对应工具

## 快速开始

### 环境要求

| 组件 | 版本 |
|------|------|
| Node.js | >= 22.19.0 (nvm) |
| Python | 3.10-3.12 (conda `zzq`) |
| CUDA | BGE-VL-large 需要 GPU |

### 1. 安装 Python 依赖

```bash
conda activate zzq
pip install -r requirements.txt
```

> 各子服务也有独立的 `requirements.txt`，可按需单独安装。

### 2. 启动 Python 服务

```bash
# 检测校验服务 (端口 8001)
cd services/detection-service
bash start.sh

# 数据分析服务 (端口 8002)
cd ../data-analysis-service
uvicorn app:app --host 0.0.0.0 --port 8002
```

### 3. 验证服务

```bash
# 检测服务
curl http://localhost:8001/health

# 分析服务
curl http://localhost:8002/health
```

### 4. 启动 PI Agent

```bash
npm install
npm run agent
```

Agent 启动后进入交互式 REPL，可直接用自然语言操作，例如：
- "帮我验证这张图片的检测结果"
- "分析一下误报目录的数据集质量"

## 项目结构

```
data-feedback-agent/
├── agent.ts                    # PI Agent 入口
├── tools.ts                    # 7 个 HTTP 工具封装
├── config.yaml                 # 共享配置（端口、LLM、数据路径）
├── requirements.txt            # Python 依赖（合并两个服务）
├── .pi/                        # PI Agent 配置
│   ├── settings.json           # 模型/Provider 配置
│   └── skills/                 # 3 个工作流技能
├── services/
│   ├── detection-service/      # 检测校验服务 (:8001)
│   │   ├── app.py              # FastAPI 入口
│   │   ├── verifier.py         # 核心：LLM 校验编排
│   │   ├── llm/                # LLM 客户端（OpenAI/Anthropic/Ollama）
│   │   ├── prompts/            # 校验/纠偏 Prompt 模板
│   │   └── services/           # 纠偏服务 + 可视化
│   └── data-analysis-service/  # 数据分析服务 (:8002)
│       ├── app.py              # FastAPI 入口
│       ├── pipeline.py         # 分析 Pipeline 编排
│       └── analyzers/          # 8 个分析模块
└── tests/                      # 测试
```

## 架构

```
用户 → PI Agent (GPT-5.5)
         │
         ├── verify_detection    → detection-service → YOLO API + MiMo LLM
         ├── correct_detection   → detection-service → 补充漏检 bbox + 对比图
         ├── analyze_dataset     → data-analysis-service → 类别分布统计
         └── compare_data        → data-analysis-service → 7维度分析 + LLM归因
```

- **Agent LLM** (GPT-5.5)：负责推理和工具选择
- **Service LLM** (MiMo v2.5)：负责检测校验和归因分析的结构化输出
- **BGE-VL-large**：CLIP 模型，用于图像语义 Embedding 和维度分类

## 测试

```bash
# 检测服务测试
cd services/detection-service && python test/test_service.py

# 分析服务测试
cd services/data-analysis-service && python tests/test_attribution.py

# Agent 集成测试
node tests/test_agent.ts
```

## 配置说明

编辑 `config.yaml` 修改：

| 配置项 | 说明 |
|--------|------|
| `llm.protocol` | LLM 协议：openai / anthropic / ollama |
| `llm.api_url` | LLM API 地址 |
| `llm.model` | 模型名称 |
| `detection.api_url` | YOLO 检测 API 地址 |
| `semantic.model_path` | BGE-VL-large 模型路径 |
| `data.train_dir` | 训练集路径 |
| `data.test_dir` | 测试集路径 |

## 文档

- [架构设计](ARCHITECTURE.md)
- [Windows 迁移说明](REFACTORING_WINDOWS.md)
- [检测校验流程](docx/agent_monitoring_feedback_flowchart.md)
- [数据分析报告](docx/analysis_report.md)

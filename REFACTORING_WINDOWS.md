# Windows 环境重构说明

## 概述

本文档记录了将 `data-feedback-agent` 项目从 Linux 环境迁移到 Windows 环境所做的修改。

## 路径映射

| Linux 路径 | Windows 路径 |
|-----------|-------------|
| `/home/shao/zzq` | `E:\zzq` |
| `/home/shao/zzq/训练集_500` | `E:\zzq\训练集_500` |
| `/home/shao/zzq/训练集/vehicle-13631-v18-cls9` | `E:\zzq\训练集\vehicle-13631-v18-cls9` |
| `/home/shao/zzq/误报/2025-04-01/sample` | `E:\zzq\误报\2025-04-01\sample` |
| `/home/shao/zzq/model/BGE-VL-large` | `E:\zzq\model\BGE-VL-large` |

## 修改的文件

### 1. 全局配置文件

**config.yaml**
- 修改数据路径配置
- 修改模型路径配置

### 2. Python 配置加载器

**services/detection-service/config.py**
- 修改默认配置文件路径

**services/data-analysis-service/config.py**
- 修改默认配置文件路径

**services/data-analysis-service/analyzers/semantic_analyzer.py**
- 修改默认模型路径

### 3. 启动脚本

**run.sh**
- 添加 Windows 环境下的 nvm 路径处理
- 添加注释说明使用 Git Bash 运行

**services/detection-service/start.sh**
- 添加注释说明使用 Git Bash 运行

### 4. 测试文件

**services/detection-service/test/test_service.py**
- 修改测试数据目录路径

**services/detection-service/test/evaluate_secondary_review.py**
- 修改默认参数路径

**test_attribution.py**
- 修改测试图片路径

**test_agent.ts**
- 修改测试图片路径

## 环境要求

### Windows 环境

1. **Node.js**: 使用 nvm-windows 管理 Node.js 版本
   - 安装 nvm-windows: https://github.com/coreybutler/nvm-windows
   - 安装 Node.js 22: `nvm install 22`
   - 使用 Node.js 22: `nvm use 22`

2. **Python**: 使用 conda 管理 Python 环境
   - 激活环境: `conda activate zzq`

3. **Git Bash**: 用于运行 bash 脚本
   - 安装 Git for Windows: https://git-scm.com/download/win

### 依赖安装

```bash
# Python 依赖
cd services/detection-service
pip install -r requirements.txt

cd ../data-analysis-service
pip install -r requirements.txt

# Node.js 依赖
npm install
```

## 启动服务

### 1. 启动检测校验服务 (端口 8001)

```bash
# 使用 Git Bash
cd services/detection-service
bash start.sh

# 或使用 Python
cd services/detection-service
python app.py
```

### 2. 启动数据分析服务 (端口 8002)

```bash
# 使用 Python
cd services/data-analysis-service
python app.py

# 或使用 uvicorn
cd services/data-analysis-service
uvicorn app:app --host 0.0.0.0 --port 8002
```

### 3. 启动 PI Agent

```bash
# 使用 npm
npm run agent

# 或使用 pi
npx pi run agent.ts
```

## 测试

### 测试检测服务

```bash
cd services/detection-service
python test/test_service.py
```

### 测试数据分析服务

```bash
curl http://localhost:8002/health
```

### 测试 Agent

```bash
npm run test
```

## 注意事项

1. **路径分隔符**: Windows 使用反斜杠 `\`，Python 和 Node.js 可以处理两种分隔符
2. **编码**: 确保文件使用 UTF-8 编码
3. **GPU**: 确保 CUDA 驱动已安装，PyTorch 支持 GPU
4. **模型文件**: 确保 BGE-VL-large 模型文件存在于 `E:\zzq\model\BGE-VL-large`

## 故障排除

### 问题 1: nvm 命令找不到

**解决方案**: 安装 nvm-windows 并重启终端

### 问题 2: Python 模块找不到

**解决方案**: 确保在正确的 conda 环境中，并设置了 PYTHONPATH

### 问题 3: 模型加载失败

**解决方案**: 检查模型路径是否正确，确保模型文件完整

### 问题 4: 服务启动失败

**解决方案**: 检查端口是否被占用，确保依赖已安装

## 下一步

1. 测试所有服务是否正常启动
2. 验证模型加载是否成功
3. 运行完整的测试套件
4. 更新文档中的路径引用

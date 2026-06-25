#!/bin/bash
# 启动检测校验服务
# Windows 环境: 使用 Git Bash 运行此脚本
cd "$(dirname "$0")"

# 加载 .env（如果存在）
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# 加载 config.yaml 中的配置（环境变量优先）
PORT=${PORT:-8001}

echo "============================================"
echo "  检测校验服务"
echo "============================================"
echo "  端口:     :${PORT}"
echo "  LLM协议:  $(python -c 'import yaml; c=yaml.safe_load(open("config.yaml")); print(c["llm"]["protocol"])')"
echo "  LLM模型:  $(python -c 'import yaml; c=yaml.safe_load(open("config.yaml")); p=c["llm"]["protocol"]; print(c["llm"][p]["model"])')"
echo "  检测API:  $(python -c 'import yaml; c=yaml.safe_load(open("config.yaml")); print(c["detection"]["api_url"])')"
echo "============================================"
echo ""

python app.py

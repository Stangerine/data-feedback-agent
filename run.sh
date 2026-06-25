#!/bin/bash
# 一键运行脚本 - 自动激活环境并执行命令
# 用法: ./run.sh basic / ./run.sh chat / ./run.sh tools
# Windows 环境: 使用 Git Bash 运行此脚本

set -e

# Windows 环境下的 nvm 路径
export NVM_DIR="$APPDATA/nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# 尝试使用 nvm，如果不存在则检查 node 是否可用
if command -v nvm &> /dev/null; then
    nvm use 22 > /dev/null 2>&1
else
    echo "注意: nvm 未找到，使用系统 Node.js"
fi

# 运行指定的 npm script
npm run "${1:-basic}"

#!/usr/bin/env node

/**
 * 测试 session-review.mjs 的修改总结功能
 * 运行: node scripts/test-session-review.mjs
 */

import { execSync } from "child_process";

console.log("🧪 测试 session-review.mjs 修改总结功能\n");

// 模拟一些修改
console.log("1. 创建测试文件...");
execSync('echo "// test file" > /tmp/test-review.txt', { encoding: "utf-8" });

console.log("2. 运行 session-review.mjs...");
try {
  const output = execSync("node .claude/hooks/session-review.mjs", {
    encoding: "utf-8",
    timeout: 15000,
    stdio: ["pipe", "pipe", "pipe"]
  });
  console.log(output);
} catch (error) {
  console.log("输出:", error.stdout);
  if (error.stderr) {
    console.log("错误:", error.stderr);
  }
}

console.log("\n✅ 测试完成");

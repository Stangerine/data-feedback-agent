/**
 * Agent 自动化测试 — 验证工具注册和 API 调用
 */

import { resolve } from "node:path";
import {
  createAgentSession,
  DefaultResourceLoader,
  type ExtensionAPI,
} from "@earendil-works/pi-coding-agent";

import { detectionTools } from "./tools.js";

const projectDir = import.meta.dirname;

// 扩展：注册工具
function detectionToolsExtension(pi: ExtensionAPI) {
  for (const tool of detectionTools) {
    pi.registerTool(tool);
  }
}

// 创建 Agent
const loader = new DefaultResourceLoader({
  cwd: projectDir,
  agentDir: resolve(projectDir, ".pi"),
  extensionFactories: [detectionToolsExtension],
});
await loader.reload();

const { session } = await createAgentSession({
  agentDir: resolve(projectDir, ".pi"),
  resourceLoader: loader,
});

// 订阅事件
let responseText = "";
session.subscribe((event) => {
  if (event.type === "message_update") {
    const e = event.assistantMessageEvent;
    if (e.type === "text_delta") responseText += e.delta;
    if (e.type === "text_end") {
      // 一条完整消息结束
    }
  }
  if (event.type === "tool_execution_start") {
    console.log(`  🔧 工具调用: ${event.toolName}`);
  }
});

// ── 测试 1：检查服务状态 ────────────────────────────────────

console.log("TEST 1: 检查检测校验服务状态");
console.log("-".repeat(40));
responseText = "";
await session.prompt("请调用 check_detection_service 工具检查服务状态");
console.log(`\n  Agent 回复: ${responseText.slice(0, 200)}\n`);

// ── 测试 2：直接校验 ────────────────────────────────────────

console.log("TEST 2: 直接校验（模拟检测结果）");
console.log("-".repeat(40));
responseText = "";
await session.prompt(
  `请调用 verify_direct 工具校验这张图片：E:\\zzq\\训练集\\vehicle-13631-v18-cls9\\images\\00004.jpg
检测结果：
- class_name: wajueji, confidence: 0.92, bbox: [100, 100, 400, 400]
- class_name: dazhuangji, confidence: 0.78, bbox: [50, 50, 200, 200]`
);
console.log(`\n  Agent 回复: ${responseText.slice(0, 500)}\n`);

// ── 完成 ────────────────────────────────────────────────────

console.log("=== 测试完成 ===");
session.dispose();

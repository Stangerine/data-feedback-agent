/**
 * 数据回流 Agent — 入口脚本
 *
 * 使用方式：
 *   npx pi run agent.ts
 *
 * 功能：
 *   - 自动加载 detection-review 技能
 *   - 注册检测校验工具（verify_detection / correct_detection / verify_direct / check_detection_service）
 *   - 交互式对话
 */

import { resolve } from "node:path";
import { createInterface } from "node:readline";
import {
  createAgentSession,
  DefaultResourceLoader,
  type ExtensionAPI,
} from "@earendil-works/pi-coding-agent";

import { detectionTools, analysisTools } from "./tools.js";

const projectDir = import.meta.dirname;

// ── 扩展：注册检测校验工具 ──────────────────────────────────

function detectionToolsExtension(pi: ExtensionAPI) {
  for (const tool of [...detectionTools, ...analysisTools]) {
    pi.registerTool(tool);
  }
  console.log("✅ 已注册工具: verify_detection, correct_detection, verify_direct, check_detection_service, analyze_dataset, compare_data, check_analysis_service");
}

// ── 创建 Agent ───────────────────────────────────────────────

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

// ── 交互式对话 ───────────────────────────────────────────────

const rl = createInterface({ input: process.stdin, output: process.stdout });
const ask = (q: string) => new Promise<string>((r) => rl.question(q, r));

let running = true;

session.subscribe((event) => {
  if (!running) return;
  if (event.type === "message_update") {
    const e = event.assistantMessageEvent;
    if (e.type === "text_delta") process.stdout.write(e.delta);
  }
});

console.log("=== 数据回流 Agent ===");
console.log("可用工具: verify_detection / correct_detection / verify_direct / check_detection_service / analyze_dataset / compare_data / check_analysis_service");
console.log("可用技能: detection-review / detection-correction / data-analysis");
console.log('输入 "exit" 退出\n');

while (running) {
  try {
    const input = await ask("你: ");
    const trimmed = input.trim();
    if (!trimmed || trimmed === "exit") {
      running = false;
      break;
    }
    console.log("");
    await session.prompt(trimmed);
    console.log("\n");
  } catch {
    running = false;
    break;
  }
}

rl.close();
session.dispose();
console.log("\n再见！");

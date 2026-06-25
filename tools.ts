/**
 * 检测校验工具 — 封装 detection-service 的 HTTP API
 *
 * 工具列表：
 *   verify_detection      — 完整校验（检测API + LLM）
 *   correct_detection     — 完整纠正（检测API + LLM纠错 + 可视化）
 *   verify_direct         — 直接校验（跳过检测API）
 *   check_detection_service — 服务健康检查
 */

import { Type } from "@earendil-works/pi-ai";
import { defineTool } from "@earendil-works/pi-coding-agent";

const SERVICE_URL = process.env.DETECTION_SERVICE_URL || "http://localhost:8001";

// ── 通用请求封装 ─────────────────────────────────────────────

async function apiRequest(path: string, method: string, body?: unknown): Promise<unknown> {
  const url = `${SERVICE_URL}${path}`;
  const resp = await fetch(url, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
    signal: AbortSignal.timeout(300_000), // 5 分钟超时
  });

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`HTTP ${resp.status}: ${text}`);
  }
  return resp.json();
}

// ── 工具 1：完整校验 ────────────────────────────────────────

export const verifyDetectionTool = defineTool({
  name: "verify_detection",
  label: "检测校验",
  description: "对图片进行完整的目标检测二次校验。自动调用线上YOLO检测API获取小模型结果，再用大模型进行二次校验，输出结构化的误报/漏报分析。",
  parameters: Type.Object({
    image_path: Type.String({ description: "图片的本地绝对路径，如 /data/online/001.jpg" }),
    box_threshold: Type.Optional(
      Type.Number({ description: "置信度阈值(0-1)，默认0.5" })
    ),
  }),
  async execute(_toolCallId, params) {
    try {
      const body: Record<string, unknown> = { image_path: params.image_path };
      if (params.box_threshold != null) body.box_threshold = params.box_threshold;

      const result = await apiRequest("/api/verify", "POST", body);
      return {
        content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
        details: result,
      };
    } catch (e) {
      const msg = `校验失败: ${e instanceof Error ? e.message : e}`;
      return {
        content: [{ type: "text", text: msg }],
        details: { error: msg },
      };
    }
  },
});

// ── 工具 1.5：完整纠正 ──────────────────────────────────────

export const correctDetectionTool = defineTool({
  name: "correct_detection",
  label: "检测纠正",
  description: "对图片进行目标检测纠正。自动调用线上YOLO检测API获取小模型结果，再用多模态大模型修正误报类别、为漏报目标补充bbox，并保存小模型标注图和大模型纠正图用于可视化对比。",
  parameters: Type.Object({
    image_path: Type.String({ description: "图片的本地绝对路径，如 /data/online/001.jpg" }),
    box_threshold: Type.Optional(
      Type.Number({ description: "置信度阈值(0-1)，默认0.5" })
    ),
  }),
  async execute(_toolCallId, params) {
    try {
      const body: Record<string, unknown> = { image_path: params.image_path };
      if (params.box_threshold != null) body.box_threshold = params.box_threshold;

      const result = await apiRequest("/api/correct", "POST", body);
      const data = result as Record<string, unknown>;
      const artifacts = (data.artifacts || {}) as Record<string, unknown>;
      const text = [
        `纠正完成: ${data.image_path}`,
        `小模型目标数: ${(data.detections as unknown[] | undefined)?.length ?? 0}`,
        `纠正后目标数: ${(data.corrected_detections as unknown[] | undefined)?.length ?? 0}`,
        `输出目录: ${artifacts.dir || ""}`,
        `小模型标注图: ${artifacts.small_model_image || ""}`,
        `大模型纠正图: ${artifacts.corrected_image || ""}`,
        `结果JSON: ${artifacts.result_json || ""}`,
      ].join("\n");
      return {
        content: [{ type: "text", text }],
        details: result,
      };
    } catch (e) {
      const msg = `纠正失败: ${e instanceof Error ? e.message : e}`;
      return {
        content: [{ type: "text", text: msg }],
        details: { error: msg },
      };
    }
  },
});

// ── 工具 2：直接校验 ────────────────────────────────────────

export const verifyDirectTool = defineTool({
  name: "verify_direct",
  label: "直接校验",
  description: "跳过检测API，直接传入已有的检测结果进行大模型二次校验。适用于已有YOLO检测结果、只想做大模型校验的场景。",
  parameters: Type.Object({
    image_path: Type.String({ description: "图片的本地绝对路径" }),
    detections: Type.Array(
      Type.Object({
        class_id: Type.Number({ description: "类别ID" }),
        class_name: Type.String({ description: "类别英文名，如 wajueji, chanche" }),
        confidence: Type.Number({ description: "置信度 0-1" }),
        bbox: Type.Array(Type.Number({ description: "坐标" }), {
          description: "[x1, y1, x2, y2] 格式的检测框",
        }),
      }),
      { description: "检测结果列表" }
    ),
  }),
  async execute(_toolCallId, params) {
    try {
      const result = await apiRequest("/api/verify_direct", "POST", {
        image_path: params.image_path,
        detections: params.detections,
      });
      return {
        content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
        details: result,
      };
    } catch (e) {
      const msg = `校验失败: ${e instanceof Error ? e.message : e}`;
      return {
        content: [{ type: "text", text: msg }],
        details: { error: msg },
      };
    }
  },
});

// ── 工具 3：健康检查 ────────────────────────────────────────

export const checkServiceTool = defineTool({
  name: "check_detection_service",
  label: "服务状态检查",
  description: "检查检测校验服务是否正常运行，返回服务状态、LLM协议、模型信息。",
  parameters: Type.Object({}),
  async execute() {
    try {
      const result = await apiRequest("/health", "GET");
      const data = result as Record<string, unknown>;
      const text = [
        `状态: ${data.status}`,
        `LLM协议: ${data.llm_protocol}`,
        `LLM模型: ${data.llm_model}`,
        `检测API: ${data.detection_api}`,
      ].join("\n");
      return {
        content: [{ type: "text", text }],
        details: result,
      };
    } catch (e) {
      const msg = `服务不可用: ${e instanceof Error ? e.message : e}`;
      return {
        content: [{ type: "text", text: msg }],
        details: { error: msg },
      };
    }
  },
});

// ── 数据分析工具 ────────────────────────────────────────────

const ANALYSIS_URL = process.env.DATA_ANALYSIS_URL || "http://localhost:8002";

async function analysisRequest(path: string, method: string, body?: unknown): Promise<unknown> {
  const url = `${ANALYSIS_URL}${path}`;
  const resp = await fetch(url, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
    signal: AbortSignal.timeout(600_000), // 10 分钟超时 (分析可能耗时)
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`HTTP ${resp.status}: ${text}`);
  }
  return resp.json();
}

// ── 工具 4：数据集画像 ──────────────────────────────────────

export const analyzeDatasetTool = defineTool({
  name: "analyze_dataset",
  label: "数据集画像",
  description: "对指定数据集进行画像分析，统计类别分布、边界框特征、图片尺寸等信息。用于了解训练数据或测试数据的整体特征。",
  parameters: Type.Object({
    data_dir: Type.String({ description: "数据集目录路径，需包含 images/ 子目录" }),
  }),
  async execute(_toolCallId, params) {
    try {
      const result = await analysisRequest("/api/profile", "POST", {
        data_dir: params.data_dir,
      });
      const data = (result as Record<string, unknown>).data as Record<string, unknown>;
      const classDist = (data.class_distribution as Array<Record<string, unknown>> || [])
        .map(c => `  ${c.class_name}: ${c.count} (${c.percentage}%)`)
        .join("\n");
      const text = [
        `数据集: ${data.data_dir}`,
        `图片数: ${data.image_count}`,
        `目标总数: ${data.total_objects}`,
        `类别分布:\n${classDist}`,
        `边界框均面积比: ${(data.bbox_stats as Record<string, unknown>)?.mean_area_ratio}`,
      ].join("\n");
      return {
        content: [{ type: "text", text }],
        details: result,
      };
    } catch (e) {
      const msg = `画像分析失败: ${e instanceof Error ? e.message : e}`;
      return {
        content: [{ type: "text", text: msg }],
        details: { error: msg },
      };
    }
  },
});

// ── 工具 5：完整对比分析 ────────────────────────────────────

export const compareDataTool = defineTool({
  name: "compare_data",
  label: "数据对比分析",
  description: "对训练数据和测试(误报)数据进行多维度对比分析，包括类别覆盖、Embedding分布、图像质量、空间特征、LLM归因，并给出回流价值评分和建议。这是核心分析工具。",
  parameters: Type.Object({
    training_dir: Type.Optional(Type.String({ description: "训练数据目录，默认使用配置中的训练集路径" })),
    test_dir: Type.Optional(Type.String({ description: "测试(误报)数据目录，默认使用配置中的误报路径" })),
    skip_llm: Type.Optional(Type.Boolean({ description: "是否跳过LLM归因分析(快速模式)，默认false" })),
    skip_embedding: Type.Optional(Type.Boolean({ description: "是否跳过Embedding分析，默认false" })),
  }),
  async execute(_toolCallId, params) {
    try {
      const body: Record<string, unknown> = {};
      if (params.training_dir) body.training_dir = params.training_dir;
      if (params.test_dir) body.test_dir = params.test_dir;
      if (params.skip_llm) body.skip_llm = true;
      if (params.skip_embedding) body.skip_embedding = true;

      const result = await analysisRequest("/api/compare", "POST", body);
      const data = (result as Record<string, unknown>).data as Record<string, unknown>;
      const summary = data.summary as Record<string, unknown>;

      // 构建摘要文本
      const recommended = (summary.recommended_for_feedback as string[] || []);
      const review = (summary.needs_review as string[] || []);

      const text = [
        `=== 数据对比分析报告 ===`,
        ``,
        `训练集: ${data.training_dir} (${data.train_image_count} 张)`,
        `测试集: ${data.test_dir} (${data.test_image_count} 张)`,
        ``,
        `--- 回流建议 ---`,
        `推荐回流 (${summary.high_value_count} 张): ${recommended.join(", ") || "无"}`,
        `建议复核 (${summary.medium_value_count} 张): ${review.join(", ") || "无"}`,
        `不建议回流: ${summary.low_value_count} 张`,
        ``,
        `耗时: ${summary.duration_seconds}s`,
        `完整报告: ${data.report_path || "见服务端"}`,
      ].join("\n");

      return {
        content: [{ type: "text", text }],
        details: result,
      };
    } catch (e) {
      const msg = `对比分析失败: ${e instanceof Error ? e.message : e}`;
      return {
        content: [{ type: "text", text: msg }],
        details: { error: msg },
      };
    }
  },
});

// ── 工具 6：检查分析服务 ────────────────────────────────────

export const checkAnalysisServiceTool = defineTool({
  name: "check_analysis_service",
  label: "分析服务状态",
  description: "检查数据分析服务是否正常运行，返回服务状态、模型信息。",
  parameters: Type.Object({}),
  async execute() {
    try {
      const result = await analysisRequest("/health", "GET");
      const data = result as Record<string, unknown>;
      const text = [
        `状态: ${data.status}`,
        `Embedding模型: ${data.embedding_model}`,
        `质量模型: ${data.quality_model}`,
        `LLM模型: ${data.llm_model}`,
      ].join("\n");
      return {
        content: [{ type: "text", text }],
        details: result,
      };
    } catch (e) {
      const msg = `分析服务不可用: ${e instanceof Error ? e.message : e}`;
      return {
        content: [{ type: "text", text: msg }],
        details: { error: msg },
      };
    }
  },
});

/** 导出所有工具 */
export const detectionTools = [
  verifyDetectionTool,
  correctDetectionTool,
  verifyDirectTool,
  checkServiceTool,
];

export const analysisTools = [
  analyzeDatasetTool,
  compareDataTool,
  checkAnalysisServiceTool,
];

/** 导出全部工具 */
export const allTools = [...detectionTools, ...analysisTools];

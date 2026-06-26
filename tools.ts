/**
 * 检测校验工具 — 封装 detection-service 的 HTTP API
 *
 * 工具列表：
 *   verify_detection      — 完整校验（检测API + LLM）
 *   correct_detection     — 完整纠正（检测API + LLM纠错 + 可视化）
 *   verify_direct         — 直接校验（跳过检测API）
 *   check_detection_service — 服务健康检查
 *
 * 数据分析工具 — 封装 data-analysis-service 的 HTTP API
 *   analyze_dataset       — 数据集画像
 *   analyze_single        — 单图归因分析
 *   analyze_batch         — 批量归因分析
 *   export_analysis       — 导出分析结果
 *   check_analysis_service — 分析服务状态检查
 */

import { Type } from "@earendil-works/pi-ai";
import { defineTool } from "@earendil-works/pi-coding-agent";

const DETECTION_URL = process.env.DETECTION_SERVICE_URL || "http://localhost:8001";
const ANALYSIS_URL = process.env.DATA_ANALYSIS_URL || "http://localhost:8002";

// ── 通用请求封装 ─────────────────────────────────────────────

async function serviceRequest(baseUrl: string, path: string, method: string, body?: unknown, timeout = 300_000): Promise<unknown> {
  const url = `${baseUrl}${path}`;
  const resp = await fetch(url, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
    signal: AbortSignal.timeout(timeout),
  });

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`HTTP ${resp.status}: ${text}`);
  }
  return resp.json();
}

async function detectionRequest(path: string, method: string, body?: unknown): Promise<unknown> {
  return serviceRequest(DETECTION_URL, path, method, body, 300_000);
}

async function analysisRequest(path: string, method: string, body?: unknown): Promise<unknown> {
  return serviceRequest(ANALYSIS_URL, path, method, body, 600_000);
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

      const result = await detectionRequest("/api/verify", "POST", body);
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

      const result = await detectionRequest("/api/correct", "POST", body);
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
      const result = await detectionRequest("/api/verify_direct", "POST", {
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
      const result = await detectionRequest("/health", "GET");
      const data = result as Record<string, unknown>;
      const text = [
        `状态: ${data.status}`,
        `服务: ${data.service} v${data.version}`,
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

// ── 工具 5：单图归因分析 ────────────────────────────────────

export const analyzeSingleTool = defineTool({
  name: "analyze_single",
  label: "单图归因分析",
  description: "对单张误报/漏报图片进行多维度归因分析。结合检测结果和校验结果，从类别分布、光照、视角、模糊、天气、时间、环境等维度分析误报/漏报原因，给出是否建议回流的判断。",
  parameters: Type.Object({
    image_path: Type.String({ description: "图片的本地绝对路径" }),
    detections: Type.Optional(Type.Array(
      Type.Object({
        class_name: Type.String({ description: "类别名" }),
        confidence: Type.Number({ description: "置信度" }),
        bbox: Type.Optional(Type.Array(Type.Number(), { description: "[x1,y1,x2,y2]" })),
      }),
      { description: "当前检测结果列表" }
    )),
    false_positives: Type.Optional(Type.Array(
      Type.Object({
        class_name: Type.String({ description: "被误报的类别" }),
        confidence: Type.Number({ description: "置信度" }),
        reason: Type.Optional(Type.String({ description: "误报原因" })),
      }),
      { description: "误报列表" }
    )),
    false_negatives: Type.Optional(Type.Array(
      Type.Object({
        class_name: Type.String({ description: "漏报的类别" }),
        reason: Type.Optional(Type.String({ description: "漏报原因" })),
      }),
      { description: "漏报列表" }
    )),
    verification_result: Type.Optional(Type.Any({ description: "detection-service 的校验结果 (完整JSON)" })),
  }),
  async execute(_toolCallId, params) {
    try {
      const body: Record<string, unknown> = {
        image_path: params.image_path,
        detections: params.detections || [],
        false_positives: params.false_positives || [],
        false_negatives: params.false_negatives || [],
      };
      if (params.verification_result) body.verification_result = params.verification_result;

      const result = await analysisRequest("/api/analyze/single", "POST", body);
      const res = result as Record<string, unknown>;
      const data = res.data as Record<string, unknown>;

      const dims = (data.dimension_attributions as Array<Record<string, unknown>> || [])
        .map(d => `  - ${d.dimension}: ${d.category} (训练集占比 ${(d.train_coverage as number * 100).toFixed(1)}%, 缺口: ${d.is_gap ? "是" : "否"})`)
        .join("\n");

      const text = [
        `=== 单图归因分析 ===`,
        `图片: ${data.filename}`,
        `归因类型: ${data.attribution_type}`,
        `置信度: ${(data.confidence as number).toFixed(2)}`,
        `主因维度: ${data.main_cause_dimension}`,
        `是否建议回流: ${data.should_feedback ? "是" : "否"}`,
        ``,
        `各维度归因:\n${dims}`,
        ``,
        `回流建议: ${data.feedback_suggestion}`,
      ].join("\n");

      return {
        content: [{ type: "text", text }],
        details: result,
      };
    } catch (e) {
      const msg = `归因分析失败: ${e instanceof Error ? e.message : e}`;
      return {
        content: [{ type: "text", text: msg }],
        details: { error: msg },
      };
    }
  },
});

// ── 工具 6：批量归因分析 ────────────────────────────────────

export const analyzeBatchTool = defineTool({
  name: "analyze_batch",
  label: "批量归因分析",
  description: "对整个测试集进行批量归因分析。预计算训练集分布后，逐张分析测试图片，输出每张图片的归因类型和回流建议。首次运行需要较长时间(训练集embedding计算)。",
  parameters: Type.Object({
    training_dir: Type.Optional(Type.String({ description: "训练数据目录，默认使用配置中的训练集路径" })),
    test_dir: Type.Optional(Type.String({ description: "测试(误报)数据目录，默认使用配置中的误报路径" })),
  }),
  async execute(_toolCallId, params) {
    try {
      const body: Record<string, unknown> = {};
      if (params.training_dir) body.training_dir = params.training_dir;
      if (params.test_dir) body.test_dir = params.test_dir;

      const result = await analysisRequest("/api/analyze/batch", "POST", body);
      const res = result as Record<string, unknown>;
      const data = res.data as Record<string, unknown>;
      const summary = data.summary as Record<string, unknown>;

      const text = [
        `=== 批量归因分析报告 ===`,
        ``,
        `训练集: ${data.training_dir}`,
        `测试集: ${data.test_dir}`,
        `测试图片数: ${data.test_image_count}`,
        ``,
        `--- 统计 ---`,
        `总计: ${summary.total} 张`,
        `建议回流: ${summary.should_feedback} 张`,
        `不建议回流: ${summary.should_not_feedback} 张`,
      ].join("\n");

      return {
        content: [{ type: "text", text }],
        details: result,
      };
    } catch (e) {
      const msg = `批量分析失败: ${e instanceof Error ? e.message : e}`;
      return {
        content: [{ type: "text", text: msg }],
        details: { error: msg },
      };
    }
  },
});

// ── 工具 7：导出分析结果 ────────────────────────────────────

export const exportAnalysisTool = defineTool({
  name: "export_analysis",
  label: "导出分析结果",
  description: "导出归因分析结果，支持JSON和CSV格式，可按回流等级过滤。",
  parameters: Type.Object({
    format: Type.Optional(Type.String({ description: "导出格式: json 或 csv，默认 json" })),
    min_level: Type.Optional(Type.String({ description: "过滤等级: feedback(仅回流) / no_feedback(不回流) / all(全部)，默认 all" })),
  }),
  async execute(_toolCallId, params) {
    try {
      const body: Record<string, unknown> = {
        format: params.format || "json",
        min_level: params.min_level || "all",
      };

      const result = await analysisRequest("/api/export", "POST", body);
      const res = result as Record<string, unknown>;

      const text = [
        `导出成功`,
        `格式: ${body.format}`,
        `过滤: ${body.min_level}`,
        `记录数: ${res.filtered_count ?? res.count}`,
        `文件路径: ${res.path}`,
      ].join("\n");

      return {
        content: [{ type: "text", text }],
        details: result,
      };
    } catch (e) {
      const msg = `导出失败: ${e instanceof Error ? e.message : e}`;
      return {
        content: [{ type: "text", text: msg }],
        details: { error: msg },
      };
    }
  },
});

// ── 工具 8：检查分析服务 ────────────────────────────────────

export const checkAnalysisServiceTool = defineTool({
  name: "check_analysis_service",
  label: "分析服务状态",
  description: "检查数据分析服务是否正常运行，返回服务状态、模型信息、初始化状态。",
  parameters: Type.Object({}),
  async execute() {
    try {
      const result = await analysisRequest("/health", "GET");
      const data = result as Record<string, unknown>;
      const dataDirs = data.data_dirs as Record<string, string> || {};
      const text = [
        `状态: ${data.status}`,
        `服务: ${data.service} v${data.version}`,
        `CLIP模型: ${data.clip_model}`,
        `LLM模型: ${data.llm_model}`,
        `分析维度: ${(data.dimensions as string[] || []).join(", ")}`,
        `已初始化: ${data.initialized ? "是" : "否"}`,
        `训练集: ${dataDirs.training}`,
        `测试集: ${dataDirs.test}`,
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

/** 检测校验工具 */
export const detectionTools = [
  verifyDetectionTool,
  correctDetectionTool,
  verifyDirectTool,
  checkServiceTool,
];

/** 数据分析工具 */
export const analysisTools = [
  analyzeDatasetTool,
  analyzeSingleTool,
  analyzeBatchTool,
  exportAnalysisTool,
  checkAnalysisServiceTool,
];

/** 导出全部工具 */
export const allTools = [...detectionTools, ...analysisTools];

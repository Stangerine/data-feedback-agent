"""数据分析服务 — FastAPI 入口 (port 8002)"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import get_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ── 初始化 ──────────────────────────────────────────────

config = get_config()
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from pipeline import AnalysisPipeline
        _pipeline = AnalysisPipeline(config)
    return _pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时预计算训练集分布
    logger.info("服务启动，预计算训练集分布...")
    pipeline = get_pipeline()
    pipeline.initialize()
    logger.info("训练集分布预计算完成")
    yield
    logger.info("服务关闭")


app = FastAPI(
    title="Data Analysis Service",
    description="训练数据与测试数据对比分析服务 — 多维度归因分析",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 请求模型 ────────────────────────────────────────────

class ErrorCase(BaseModel):
    """错误案例"""
    class_name: str
    confidence: float = 0.0
    reason: str = ""


class SingleAnalyzeRequest(BaseModel):
    """单图归因分析请求"""
    image_path: str
    detections: List[Dict[str, Any]] = Field(default_factory=list)  # 当前检测结果
    false_positives: List[ErrorCase] = Field(default_factory=list)  # 误报列表
    false_negatives: List[ErrorCase] = Field(default_factory=list)  # 漏报列表
    verification_result: Optional[Dict[str, Any]] = None  # detection-service 校验结果


class BatchAnalyzeRequest(BaseModel):
    """批量分析请求"""
    training_dir: Optional[str] = None
    test_dir: Optional[str] = None


class ProfileRequest(BaseModel):
    """数据集画像请求"""
    data_dir: str


class ExportRequest(BaseModel):
    """导出请求"""
    format: str = "json"
    min_level: str = "all"  # feedback / no_feedback / all


# ── 路由 ────────────────────────────────────────────────

@app.get("/health")
async def health():
    """健康检查"""
    from config import get_llm_config
    semantic_config = config.get("semantic", {})
    llm_config = get_llm_config()
    pipeline = get_pipeline()

    return {
        "status": "running",
        "service": "data-analysis-service",
        "version": "2.0.0",
        "port": config.get("server", {}).get("data_analysis_port", 8002),
        "clip_model": semantic_config.get("model_name", ""),
        "dimensions": ["lighting", "viewpoint", "blur", "weather", "timeOfDay", "environment"],
        "llm_model": llm_config.get("openai", {}).get("model", ""),
        "initialized": pipeline._initialized,
        "data_dirs": {
            "training": config.get("data", {}).get("training_dir", ""),
            "test": config.get("data", {}).get("test_dir", ""),
        },
    }


@app.post("/api/profile")
async def profile(request: ProfileRequest):
    """数据集画像 — 统计类分布、bbox、图片尺寸等"""
    try:
        from data_loader import profile_dataset
        result = profile_dataset(request.data_dir)
        result.pop("annotations", None)
        return {"success": True, "data": result}
    except Exception as e:
        logger.error(f"数据集画像失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze/single")
async def analyze_single(request: SingleAnalyzeRequest):
    """单图归因分析"""
    try:
        pipeline = get_pipeline()

        if not pipeline._initialized:
            pipeline.initialize()

        result = pipeline.analyze_single(
            image_path=request.image_path,
            detections=request.detections,
            false_positives=[fp.model_dump() for fp in request.false_positives],
            false_negatives=[fn.model_dump() for fn in request.false_negatives],
            verification_result=request.verification_result,
        )

        return {"success": True, "data": result}
    except Exception as e:
        logger.error(f"单图归因分析失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze/batch")
async def analyze_batch(request: BatchAnalyzeRequest):
    """批量归因分析"""
    try:
        pipeline = get_pipeline()
        result = pipeline.run_full(
            training_dir=request.training_dir,
            test_dir=request.test_dir,
        )
        return {"success": True, "data": result}
    except Exception as e:
        logger.error(f"批量分析失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/export")
async def export(request: ExportRequest):
    """导出分析结果"""
    try:
        output_dir = config.get("output", {}).get("dir", "./analysis_results")
        if not os.path.exists(output_dir):
            raise HTTPException(status_code=404, detail="没有分析结果")

        files = sorted([
            f for f in os.listdir(output_dir)
            if f.startswith("analysis_") and f.endswith(".json")
        ], reverse=True)

        if not files:
            raise HTTPException(status_code=404, detail="没有分析结果")

        latest = os.path.join(output_dir, files[0])
        with open(latest, "r", encoding="utf-8") as f:
            report = json.load(f)

        results = report.get("attribution_results", [])
        if request.min_level == "feedback":
            filtered = [r for r in results if r.get("should_feedback", False)]
        elif request.min_level == "no_feedback":
            filtered = [r for r in results if not r.get("should_feedback", False)]
        else:
            filtered = results

        if request.format == "csv":
            import csv
            csv_path = latest.replace(".json", f"_export_{request.min_level}.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "filename", "attribution_type", "confidence",
                    "main_cause_dimension", "should_feedback",
                    "feedback_suggestion", "reasoning",
                ])
                for r in filtered:
                    writer.writerow([
                        r.get("filename", ""),
                        r.get("attribution_type", ""),
                        r.get("confidence", 0),
                        r.get("main_cause_dimension", ""),
                        r.get("should_feedback", False),
                        r.get("feedback_suggestion", ""),
                        r.get("reasoning", ""),
                    ])
            return {"success": True, "path": csv_path, "count": len(filtered)}

        return {
            "success": True,
            "path": latest,
            "filtered_count": len(filtered),
            "data": filtered,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"导出失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── 启动 ────────────────────────────────────────────────

if __name__ == "__main__":
    host = config.get("server", {}).get("host", "0.0.0.0")
    port = config.get("server", {}).get("data_analysis_port", 8002)
    logger.info(f"Starting Data Analysis Service v2.0.0 on {host}:{port}")
    uvicorn.run(app, host=host, port=port)

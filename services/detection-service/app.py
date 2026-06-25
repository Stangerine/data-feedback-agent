"""
检测校验 API 服务 — FastAPI 入口

功能：小模型检测结果 → 大模型二次校验 → 结构化输出误报/漏报

API:
  POST /api/verify          — 单图校验（调检测API + LLM校验）
  POST /api/correct         — 单图纠正（调检测API + LLM纠错 + 可视化）
  POST /api/verify_direct   — 直接校验（跳过检测API）
  POST /api/verify_batch    — 批量校验
  GET  /health              — 健康检查
"""

import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException

from config import get_config
from detection import DetectionClient
from schemas import (
    CorrectRequest, VerifyRequest, VerifyDirectRequest, VerifyBatchRequest,
)
from services.correction_service import CorrectionService
from verifier import Verifier

app = FastAPI(
    title="检测校验服务",
    description="小模型检测结果的大模型二次校验，输出结构化误报/漏报分析",
    version="1.0.0",
)

# ── 全局实例（懒加载） ───────────────────────────────────────

_detection_client: DetectionClient | None = None
_verifier: Verifier | None = None
_correction_service: CorrectionService | None = None


def get_detection_client() -> DetectionClient:
    global _detection_client
    if _detection_client is None:
        cfg = get_config().detection
        _detection_client = DetectionClient(
            api_url=cfg.api_url,
            model_id=cfg.model_id,
            box_threshold=cfg.box_threshold,
            timeout=cfg.timeout,
        )
    return _detection_client


def get_verifier() -> Verifier:
    global _verifier
    if _verifier is None:
        _verifier = Verifier()
    return _verifier


def get_correction_service() -> CorrectionService:
    global _correction_service
    if _correction_service is None:
        _correction_service = CorrectionService()
    return _correction_service


# ── API 路由 ──────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """健康检查"""
    cfg = get_config()
    return {
        "status": "ok",
        "service": "detection-service",
        "version": "1.0.0",
        "llm_api_url": cfg.llm.api_url,
        "llm_model": cfg.llm.model,
        "detection_api": cfg.detection.api_url,
    }


@app.post("/api/verify")
def api_verify(req: VerifyRequest):
    """
    单图校验：调检测 API → 大模型二次校验

    流程：
    1. 调用线上 YOLO 检测 API 获取小模型结果
    2. 将图片 + 检测结果发给大模型
    3. 大模型输出结构化误报/漏报分析
    """
    if not os.path.exists(req.image_path):
        raise HTTPException(status_code=400, detail=f"图片不存在: {req.image_path}")

    cfg = get_config()
    threshold = req.box_threshold if req.box_threshold is not None else cfg.detection.box_threshold

    # 1. 检测
    try:
        detections = get_detection_client().detect(req.image_path, threshold)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"检测 API 失败: {e}")

    # 2. 校验
    gt = [g.model_dump() for g in req.ground_truth] if req.ground_truth else None
    try:
        result = get_verifier().verify(req.image_path, detections, gt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"大模型校验失败: {e}")

    return {
        "image_path": req.image_path,
        "detections": detections,
        "verification": result,
    }


@app.post("/api/correct")
def api_correct(req: CorrectRequest):
    """
    单图纠正：调检测 API → 大模型纠错 → 保存可视化结果

    流程：
    1. 调用线上 YOLO 检测 API 获取小模型结果
    2. 大模型修正误报类别，并为漏报目标输出 bbox
    3. 保存小模型标注图和大模型纠正图
    """
    if not os.path.exists(req.image_path):
        raise HTTPException(status_code=400, detail=f"图片不存在: {req.image_path}")

    cfg = get_config()
    threshold = req.box_threshold if req.box_threshold is not None else cfg.detection.box_threshold

    try:
        detections = get_detection_client().detect(req.image_path, threshold)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"检测 API 失败: {e}")

    try:
        verification = get_verifier().verify(req.image_path, detections)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"大模型校验失败: {e}")

    try:
        return get_correction_service().correct(req.image_path, detections, verification)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"大模型纠正失败: {e}")


@app.post("/api/verify_direct")
def api_verify_direct(req: VerifyDirectRequest):
    """
    直接校验：跳过检测 API，传入已有检测结果做二次校验

    适用：已有检测结果、本地调试、不想调远程 API
    """
    if not os.path.exists(req.image_path):
        raise HTTPException(status_code=400, detail=f"图片不存在: {req.image_path}")

    detections = [d.model_dump() for d in req.detections]
    gt = [g.model_dump() for g in req.ground_truth] if req.ground_truth else None

    try:
        result = get_verifier().verify(req.image_path, detections, gt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"大模型校验失败: {e}")

    return {
        "image_path": req.image_path,
        "detections": detections,
        "verification": result,
    }


@app.post("/api/verify_batch")
def api_verify_batch(req: VerifyBatchRequest):
    """
    批量校验：遍历目录下所有图片，逐张校验

    返回：每张图片结果 + 汇总统计
    """
    image_dir = Path(req.image_dir)
    if not image_dir.exists():
        raise HTTPException(status_code=400, detail=f"目录不存在: {req.image_dir}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = sorted([f for f in image_dir.iterdir() if f.suffix.lower() in exts])
    if req.limit:
        image_files = image_files[:req.limit]
    if not image_files:
        raise HTTPException(status_code=400, detail="目录下没有图片")

    cfg = get_config()
    threshold = req.box_threshold if req.box_threshold is not None else cfg.detection.box_threshold
    det_client = get_detection_client()
    verifier = get_verifier()

    results = []
    stats = {
        "total": len(image_files), "success": 0, "failed": 0,
        "fp_total": 0, "md_total": 0,
    }

    for i, img_file in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] {img_file.name} ...", end=" ", flush=True)
        try:
            detections = det_client.detect(str(img_file), threshold)
            result = verifier.verify(str(img_file), detections)
            result["detections"] = detections

            if result["success"]:
                stats["success"] += 1
                data = result["data"]
                stats["fp_total"] += len(data.get("false_positives", []))
                stats["md_total"] += len(data.get("missed_detections", []))
                print(f"OK fp={len(data.get('false_positives', []))} "
                      f"md={len(data.get('missed_detections', []))}")
            else:
                stats["failed"] += 1
                print(f"FAIL: {result.get('error')}")

            results.append(result)

        except Exception as e:
            stats["failed"] += 1
            results.append({"success": False, "error": str(e), "image_path": str(img_file)})
            print(f"ERROR: {e}")

    return {"image_dir": req.image_dir, "stats": stats, "results": results}


# ── 启动 ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    cfg = get_config()
    uvicorn.run(app, host=cfg.server.host, port=cfg.server.detection_service_port)

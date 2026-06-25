"""Detection correction service."""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from config import get_config
from llm import create_llm_client
from prompts.correction import BBOX_CORRECTION_SYSTEM_PROMPT, build_missed_bbox_prompt
from scenarios.engineering_vehicles import CLASS_IDS, VEHICLE_CLASSES
from tools.function_calling import BBOX_CORRECTION_TOOL

from .visualization import render_detections

MIN_ADDED_CONFIDENCE = 0.5
MIN_ADDED_AREA_RATIO = 0.001
MAX_EXISTING_IOU_FOR_MISSED = 0.5


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def normalized_bbox_to_pixels(
    bbox: dict[str, float], width: int, height: int
) -> list[int] | None:
    x1 = _clamp(bbox.get("x_min", 0))
    y1 = _clamp(bbox.get("y_min", 0))
    x2 = _clamp(bbox.get("x_max", 0))
    y2 = _clamp(bbox.get("y_max", 0))
    if x2 <= x1 or y2 <= y1:
        return None
    return [
        round(x1 * width),
        round(y1 * height),
        round(x2 * width),
        round(y2 * height),
    ]


def _class_id(class_name: str) -> int:
    return CLASS_IDS.get(normalize_class_name(class_name), -1)


def _area(bbox: list[float]) -> float:
    if len(bbox) != 4:
        return 0.0
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    intersection = _area([ix1, iy1, ix2, iy2])
    if intersection <= 0:
        return 0.0
    union = _area(a) + _area(b) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def normalize_class_name(value: str | None) -> str:
    """Normalize LLM class labels to canonical English class names."""
    if not value:
        return "unknown"

    text = str(value).strip()
    if text in VEHICLE_CLASSES:
        return text

    for match in re.findall(r"[（(]([a-z_]+)[）)]", text):
        if match in VEHICLE_CLASSES:
            return match

    for class_name in VEHICLE_CLASSES:
        if class_name in text:
            return class_name

    for class_name, class_name_cn in VEHICLE_CLASSES.items():
        if class_name_cn and class_name_cn in text:
            return class_name

    return text


def build_correction_from_verification(
    verification: dict[str, Any],
    detections: list[dict],
    missed_bbox_result: dict[str, Any] | None,
) -> dict[str, Any]:
    data = verification.get("data") or {}
    false_positives = data.get("false_positives") or []
    corrections = []

    for fp in false_positives:
        index = int(fp.get("detection_index", 0))
        if index < 1 or index > len(detections):
            continue
        actual_class = normalize_class_name(
            fp.get("actual_class") or detections[index - 1].get("class_name")
        )
        corrections.append(
            {
                "detection_index": index,
                "reported_class": fp.get("reported_class")
                or detections[index - 1].get("class_name", "unknown"),
                "corrected_class": actual_class,
                "corrected_class_cn": VEHICLE_CLASSES.get(actual_class, ""),
                "confidence": fp.get("confidence", 0),
                "reason": fp.get("reason", "基于校验结果修正类别"),
            }
        )

    missed_detections = data.get("missed_detections") or []
    missed_by_index = {idx: item for idx, item in enumerate(missed_detections, 1)}
    missed_result = missed_bbox_result or {}
    missed_corrections = []
    filtered_missed_corrections = []
    for item in missed_result.get("missed_detection_corrections", []):
        index = int(item.get("missed_index", 0))
        verified = missed_by_index.get(index)
        if verified is None:
            filtered_missed_corrections.append(
                {"item": item, "reason": "missed_index 不在校验漏报列表中"}
            )
            continue
        class_name = normalize_class_name(
            verified.get("actual_class") or item.get("class_name", "unknown")
        )
        corrected_item = dict(item)
        corrected_item["missed_index"] = index
        corrected_item["class_name"] = class_name
        corrected_item["class_name_cn"] = (
            verified.get("actual_class_cn") or VEHICLE_CLASSES.get(class_name, "")
        )
        missed_corrections.append(corrected_item)

    return {
        "false_positive_corrections": corrections,
        "missed_detection_corrections": missed_corrections,
        "filtered_missed_detection_corrections": filtered_missed_corrections,
        "summary": _build_correction_summary(
            len(corrections),
            len(missed_corrections),
            missed_result.get("summary", ""),
        ),
    }


def _build_correction_summary(fp_count: int, md_count: int, bbox_summary: str) -> str:
    parts = [f"基于校验结果修正 {fp_count} 个误报类别，补充 {md_count} 个漏报目标 bbox。"]
    if bbox_summary:
        parts.append(bbox_summary)
    return " ".join(parts)


def build_corrected_detections(
    detections: list[dict], correction: dict[str, Any], width: int, height: int
) -> list[dict]:
    corrected = []
    fp_by_index = {
        int(item.get("detection_index", 0)): item
        for item in correction.get("false_positive_corrections", [])
    }

    for idx, det in enumerate(detections, 1):
        item = dict(det)
        item["source"] = "small_model"
        if idx in fp_by_index:
            fp = fp_by_index[idx]
            new_class = normalize_class_name(
                fp.get("corrected_class", item.get("class_name", "unknown"))
            )
            item["original_class_name"] = item.get("class_name", "unknown")
            item["original_class_id"] = item.get("class_id", -1)
            item["class_name"] = new_class
            item["class_name_cn"] = fp.get("corrected_class_cn") or VEHICLE_CLASSES.get(
                new_class, ""
            )
            item["class_id"] = _class_id(new_class)
            item["correction_confidence"] = fp.get("confidence", 0)
            item["correction_reason"] = fp.get("reason", "")
            item["source"] = "llm_corrected"
        corrected.append(item)

    for md in correction.get("missed_detection_corrections", []):
        bbox = normalized_bbox_to_pixels(
            md.get("bbox_normalized", {}), width=width, height=height
        )
        if bbox is None:
            _record_filtered_missed(correction, md, "bbox 坐标无效")
            continue

        confidence = float(md.get("confidence", 0) or 0)
        if confidence < MIN_ADDED_CONFIDENCE:
            _record_filtered_missed(correction, md, "漏报 bbox 置信度过低")
            continue

        area_ratio = _area(bbox) / float(width * height)
        if area_ratio < MIN_ADDED_AREA_RATIO:
            _record_filtered_missed(correction, md, "漏报 bbox 面积过小")
            continue

        existing_boxes = [
            [float(v) for v in item.get("bbox", [])]
            for item in corrected
            if len(item.get("bbox", [])) == 4
        ]
        if any(_iou(bbox, existing) >= MAX_EXISTING_IOU_FOR_MISSED for existing in existing_boxes):
            _record_filtered_missed(correction, md, "漏报 bbox 与已有检测框高度重叠")
            continue
        class_name = normalize_class_name(md.get("class_name", "unknown"))
        corrected.append(
            {
                "class_id": _class_id(class_name),
                "class_name": class_name,
                "class_name_cn": md.get("class_name_cn")
                or VEHICLE_CLASSES.get(class_name, ""),
                "confidence": confidence,
                "bbox": bbox,
                "description": md.get("description", ""),
                "source": "llm_added",
            }
        )

    return corrected


def _record_filtered_missed(correction: dict[str, Any], item: dict, reason: str) -> None:
    correction.setdefault("filtered_missed_detection_corrections", []).append(
        {"item": item, "reason": reason}
    )


class CorrectionService:
    def __init__(self, llm_client=None, output_dir: str | Path = "correction_results"):
        cfg = get_config()
        self.output_dir = Path(output_dir)
        self.llm = llm_client
        if self.llm is None:
            protocol = cfg.llm.protocol
            provider_cfg = getattr(cfg.llm, protocol)
            self.llm = create_llm_client(
                protocol=protocol,
                model=provider_cfg.model,
                api_url=provider_cfg.api_url,
                api_key=provider_cfg.api_key,
                timeout=cfg.llm.timeout,
                temperature=cfg.llm.temperature,
            )

    def correct(
        self,
        image_path: str,
        detections: list[dict],
        verification: dict[str, Any],
    ) -> dict[str, Any]:
        started = time.time()
        image = Image.open(image_path).convert("RGB")
        if not verification.get("success"):
            raise RuntimeError(verification.get("error") or "verification failed")

        missed_detections = (verification.get("data") or {}).get("missed_detections") or []
        missed_bbox_result: dict[str, Any] = {
            "missed_detection_corrections": [],
            "summary": "校验结果无漏报目标，无需补框。",
        }
        if missed_detections:
            verification_data = verification.get("data") or {}
            prompt = build_missed_bbox_prompt(
                detections, verification_data, image.width, image.height
            )
            response = self.llm.chat(
                system_prompt=BBOX_CORRECTION_SYSTEM_PROMPT,
                user_prompt=prompt,
                image=image,
                tools=[BBOX_CORRECTION_TOOL],
            )
            if not response.get("success"):
                raise RuntimeError(response.get("error") or "LLM bbox correction failed")

            tool_result = response.get("tool_call")
            if tool_result is None:
                content = response.get("content") or ""
                try:
                    tool_result = json.loads(content)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"LLM bbox correction returned invalid JSON: {exc}"
                    ) from exc
            missed_bbox_result = tool_result

        correction = build_correction_from_verification(
            verification, detections, missed_bbox_result
        )

        corrected = build_corrected_detections(
            detections, correction, width=image.width, height=image.height
        )
        artifacts = self._save_artifacts(
            image_path, detections, corrected, correction, verification
        )

        return {
            "success": True,
            "image_path": image_path,
            "image_size": {"width": image.width, "height": image.height},
            "detections": detections,
            "verification": verification,
            "correction": correction,
            "corrected_detections": corrected,
            "artifacts": artifacts,
            "duration_ms": int((time.time() - started) * 1000),
        }

    def _save_artifacts(
        self,
        image_path: str,
        detections: list[dict],
        corrected: list[dict],
        correction: dict[str, Any],
        verification: dict[str, Any],
    ) -> dict[str, str]:
        stem = Path(image_path).stem
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = self.output_dir / f"{ts}_{stem}"
        out_dir.mkdir(parents=True, exist_ok=True)

        small_detections = [dict(det, source="small_model") for det in detections]
        small_image = out_dir / "small_model.jpg"
        corrected_image = out_dir / "corrected.jpg"
        result_json = out_dir / "result.json"

        render_detections(
            image_path,
            small_detections,
            str(small_image),
            "Small model detections",
        )
        render_detections(
            image_path,
            corrected,
            str(corrected_image),
            "LLM corrected detections",
        )
        result_json.write_text(
            json.dumps(
                {
                    "image_path": image_path,
                    "small_model_detections": detections,
                    "verification": verification,
                    "correction": correction,
                    "corrected_detections": corrected,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return {
            "dir": str(out_dir),
            "small_model_image": str(small_image),
            "corrected_image": str(corrected_image),
            "result_json": str(result_json),
        }

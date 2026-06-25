"""BBox correction quality tests."""

import pytest
from pydantic import ValidationError

from correction import build_missed_bbox_prompt
from schemas import VerificationResult
from services.correction_service import build_corrected_detections
from tools.function_calling import VERIFICATION_TOOL


def test_missed_detection_accepts_region_hint_and_tool_exposes_it():
    result = VerificationResult(
        false_positives=[],
        missed_detections=[
            {
                "actual_class": "wajueji",
                "actual_class_cn": "挖掘机",
                "confidence": 0.82,
                "region_hint": {
                    "x_min": 0.2,
                    "y_min": 0.3,
                    "x_max": 0.4,
                    "y_max": 0.6,
                },
                "location": "画面左侧",
                "description": "清晰可见且未被覆盖",
            }
        ],
    )

    assert result.missed_detections[0].region_hint.x_min == 0.2

    with pytest.raises(ValidationError):
        VerificationResult(
            false_positives=[],
            missed_detections=[
                {
                    "actual_class": "wajueji",
                    "confidence": 0.82,
                    "region_hint": {
                        "x_min": 0.5,
                        "y_min": 0.3,
                        "x_max": 0.4,
                        "y_max": 0.6,
                    },
                }
            ],
        )

    missed_item = (
        VERIFICATION_TOOL["function"]["parameters"]["properties"]
        ["missed_detections"]["items"]
    )
    assert "region_hint" in missed_item["properties"]


def test_missed_bbox_prompt_includes_region_hint_and_tight_box_rules():
    prompt = build_missed_bbox_prompt(
        detections=[],
        verification_data={
            "false_positives": [],
            "missed_detections": [
                {
                    "actual_class": "wajueji",
                    "actual_class_cn": "挖掘机",
                    "confidence": 0.82,
                    "region_hint": {
                        "x_min": 0.2,
                        "y_min": 0.3,
                        "x_max": 0.4,
                        "y_max": 0.6,
                    },
                    "location": "画面左侧",
                    "description": "清晰可见且未被覆盖",
                    "confidence_level": "high",
                }
            ],
        },
        width=1000,
        height=800,
    )

    assert "粗略区域(region_hint): [0.200, 0.300, 0.400, 0.600]" in prompt
    assert "优先在 region_hint 内定位" in prompt
    assert "必须同时包含车身/底盘和关键作业部件" in prompt
    assert "不要只框吊臂、桅杆、杆件" in prompt


def test_build_corrected_detections_filters_low_quality_missed_boxes():
    detections = [
        {
            "class_id": 0,
            "class_name": "wajueji",
            "confidence": 0.9,
            "bbox": [100, 100, 300, 300],
        }
    ]
    correction = {
        "false_positive_corrections": [],
        "missed_detection_corrections": [
            {
                "class_name": "chanche",
                "confidence": 0.9,
                "bbox_normalized": {
                    "x_min": 0.11,
                    "y_min": 0.11,
                    "x_max": 0.29,
                    "y_max": 0.29,
                },
                "description": "与已有框重复",
            },
            {
                "class_name": "chanche",
                "confidence": 0.4,
                "bbox_normalized": {
                    "x_min": 0.5,
                    "y_min": 0.5,
                    "x_max": 0.7,
                    "y_max": 0.7,
                },
                "description": "置信度过低",
            },
            {
                "class_name": "chanche",
                "confidence": 0.8,
                "bbox_normalized": {
                    "x_min": 0.8,
                    "y_min": 0.8,
                    "x_max": 0.805,
                    "y_max": 0.805,
                },
                "description": "面积过小",
            },
            {
                "class_name": "chanche",
                "confidence": 0.8,
                "bbox_normalized": {
                    "x_min": 0.5,
                    "y_min": 0.5,
                    "x_max": 0.7,
                    "y_max": 0.7,
                },
                "description": "有效漏报",
            },
        ],
    }

    corrected = build_corrected_detections(detections, correction, width=1000, height=1000)

    added = [item for item in corrected if item.get("source") == "llm_added"]
    assert len(added) == 1
    assert added[0]["description"] == "有效漏报"
    assert added[0]["bbox"] == [500, 500, 700, 700]

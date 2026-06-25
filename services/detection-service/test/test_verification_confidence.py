"""Verification confidence contract tests."""

import pytest
from pydantic import ValidationError

from schemas import VerificationResult
from tools.function_calling import VERIFICATION_TOOL


def test_verification_result_requires_numeric_confidence_for_fp_and_missed():
    result = VerificationResult(
        false_positives=[
            {
                "detection_index": 1,
                "reported_class": "diaoche",
                "actual_class": "wajueji",
                "confidence": 0.86,
                "reason": "检测框覆盖挖掘机而不是吊车",
            }
        ],
        missed_detections=[
            {
                "actual_class": "chanche",
                "actual_class_cn": "铲车",
                "confidence": 0.74,
                "location": "画面右侧",
                "description": "清晰可见且无检测框覆盖",
            }
        ],
    )

    assert result.false_positives[0].confidence == 0.86
    assert result.missed_detections[0].confidence == 0.74

    with pytest.raises(ValidationError):
        VerificationResult(
            false_positives=[
                {
                    "detection_index": 1,
                    "reported_class": "diaoche",
                    "actual_class": "wajueji",
                    "reason": "缺少置信度",
                }
            ],
            missed_detections=[],
        )

    with pytest.raises(ValidationError):
        VerificationResult(
            false_positives=[],
            missed_detections=[
                {
                    "actual_class": "chanche",
                    "confidence": 1.2,
                    "location": "画面右侧",
                    "description": "置信度超出范围",
                }
            ],
        )


def test_verification_function_calling_requires_confidence_fields():
    params = VERIFICATION_TOOL["function"]["parameters"]
    fp_item = params["properties"]["false_positives"]["items"]
    md_item = params["properties"]["missed_detections"]["items"]

    assert fp_item["properties"]["confidence"] == {
        "type": "number",
        "minimum": 0,
        "maximum": 1,
        "description": "误报判断置信度，0到1",
    }
    assert "confidence" in fp_item["required"]

    assert md_item["properties"]["confidence"] == {
        "type": "number",
        "minimum": 0,
        "maximum": 1,
        "description": "漏报判断置信度，0到1",
    }
    assert "confidence" in md_item["required"]

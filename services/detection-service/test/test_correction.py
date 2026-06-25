from pathlib import Path

from PIL import Image

from correction import (
    BBOX_CORRECTION_SYSTEM_PROMPT,
    CorrectionService,
    build_missed_bbox_prompt,
    build_corrected_detections,
    build_correction_from_verification,
    normalized_bbox_to_pixels,
)


def test_normalized_bbox_to_pixels_clamps_and_converts():
    bbox = {"x_min": -0.1, "y_min": 0.25, "x_max": 1.2, "y_max": 0.75}

    assert normalized_bbox_to_pixels(bbox, width=200, height=100) == [
        0,
        25,
        200,
        75,
    ]


def test_build_corrected_detections_updates_false_positive_and_adds_missed():
    detections = [
        {
            "class_id": 4,
            "class_name": "diaoche",
            "confidence": 0.89,
            "bbox": [10, 20, 110, 220],
        }
    ]
    correction = {
        "false_positive_corrections": [
            {
                "detection_index": 1,
                "corrected_class": "wajueji",
                "confidence": 0.82,
                "reason": "类别错误",
            }
        ],
        "missed_detection_corrections": [
            {
                "class_name": "chanche",
                "class_name_cn": "铲车",
                "bbox_normalized": {
                    "x_min": 0.25,
                    "y_min": 0.1,
                    "x_max": 0.5,
                    "y_max": 0.4,
                },
                "confidence": 0.7,
                "description": "漏检铲车",
            }
        ],
    }

    corrected = build_corrected_detections(detections, correction, width=400, height=300)

    assert corrected[0]["class_name"] == "wajueji"
    assert corrected[0]["class_id"] == 0
    assert corrected[0]["bbox"] == [10, 20, 110, 220]
    assert corrected[0]["source"] == "llm_corrected"
    assert corrected[0]["original_class_name"] == "diaoche"

    assert corrected[1]["class_name"] == "chanche"
    assert corrected[1]["class_id"] == 1
    assert corrected[1]["bbox"] == [100, 30, 200, 120]
    assert corrected[1]["source"] == "llm_added"


def test_build_correction_from_verification_uses_false_positive_actual_class():
    detections = [
        {
            "class_id": 4,
            "class_name": "diaoche",
            "confidence": 0.89,
            "bbox": [10, 20, 110, 220],
        }
    ]
    verification = {
        "success": True,
        "data": {
            "false_positives": [
                {
                    "detection_index": 1,
                    "reported_class": "diaoche",
                    "actual_class": "wajueji",
                    "confidence": 0.86,
                    "reason": "检测框覆盖挖掘机，非吊车",
                }
            ],
            "missed_detections": [],
        },
    }

    correction = build_correction_from_verification(verification, detections, [])

    assert correction["false_positive_corrections"] == [
        {
            "detection_index": 1,
            "reported_class": "diaoche",
            "corrected_class": "wajueji",
            "corrected_class_cn": "挖掘机",
            "confidence": 0.86,
            "reason": "检测框覆盖挖掘机，非吊车",
        }
    ]
    assert correction["missed_detection_corrections"] == []


def test_build_correction_from_verification_forces_missed_class_from_verification():
    detections = []
    verification = {
        "success": True,
        "data": {
            "false_positives": [],
            "missed_detections": [
                {
                    "actual_class": "wajueji",
                    "actual_class_cn": "挖掘机",
                    "confidence": 0.83,
                    "location": "画面左侧",
                    "description": "校验确认漏报挖掘机",
                    "confidence_level": "high",
                }
            ],
        },
    }
    missed_bbox_result = {
        "missed_detection_corrections": [
            {
                "missed_index": 1,
                "class_name": "diaoche",
                "class_name_cn": "吊车",
                "bbox_normalized": {
                    "x_min": 0.1,
                    "y_min": 0.2,
                    "x_max": 0.4,
                    "y_max": 0.6,
                },
                "confidence": 0.71,
                "description": "模型误写成吊车",
            },
            {
                "missed_index": 2,
                "class_name": "chanche",
                "bbox_normalized": {
                    "x_min": 0.5,
                    "y_min": 0.2,
                    "x_max": 0.8,
                    "y_max": 0.6,
                },
                "confidence": 0.6,
                "description": "校验列表外目标",
            },
        ],
        "summary": "bbox模型返回了类别不一致和越界序号",
    }

    correction = build_correction_from_verification(
        verification, detections, missed_bbox_result
    )

    assert correction["missed_detection_corrections"] == [
        {
            "missed_index": 1,
            "class_name": "wajueji",
            "class_name_cn": "挖掘机",
            "bbox_normalized": {
                "x_min": 0.1,
                "y_min": 0.2,
                "x_max": 0.4,
                "y_max": 0.6,
            },
            "confidence": 0.71,
            "description": "模型误写成吊车",
        }
    ]


def test_missed_bbox_prompt_is_verification_driven_and_not_a_second_review():
    detections = [
        {
            "class_id": 4,
            "class_name": "diaoche",
            "confidence": 0.89,
            "bbox": [10, 20, 110, 90],
        }
    ]
    verification_data = {
        "false_positives": [
            {
                "detection_index": 1,
                "reported_class": "diaoche",
                "actual_class": "wajueji",
                "confidence": 0.88,
                "reason": "检测框覆盖挖掘机，类别不是吊车",
            }
        ],
        "missed_detections": [
            {
                "actual_class": "dazhuangji",
                "actual_class_cn": "打桩机",
                "confidence": 0.76,
                "location": "画面中部",
                "description": "竖直桅杆清晰可见",
                "confidence_level": "high",
            }
        ],
        "overall_assessment": {
            "summary": "第1个检测框是类别误报，另有1个确认漏报。",
        },
    }

    prompt = build_missed_bbox_prompt(detections, verification_data, 200, 100)

    assert "校验结果是唯一事实来源" in BBOX_CORRECTION_SYSTEM_PROMPT
    assert "不是第二次审核" in BBOX_CORRECTION_SYSTEM_PROMPT
    assert "不得推翻" in BBOX_CORRECTION_SYSTEM_PROMPT
    assert "误报/类别修正由程序根据 false_positives 自动完成" in prompt
    assert "false_positives（只作为避让上下文" in prompt
    assert "diaoche -> wajueji" in prompt
    assert "missed_index=1" in prompt
    assert "必须等于 dazhuangji" in prompt
    assert "校验置信度: 0.76" in prompt
    assert "类别定位特征" in prompt
    assert "wajueji/挖掘机" in prompt
    assert "挖臂" in prompt
    assert "铲斗" in prompt
    assert "diaoche/吊车" in prompt
    assert "柔性钢丝绳" in prompt
    assert "吊钩" in prompt
    assert "dazhuangji/打桩机" in prompt
    assert "刚性桅杆" in prompt
    assert "chanche/铲车" in prompt
    assert "大容量开口铲斗" in prompt


def test_correction_service_bases_correction_on_verification_result(tmp_path):
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (200, 100), color="white").save(image_path)

    class FakeLLM:
        def chat(self, **kwargs):
            assert "校验结果确认的漏报目标" in kwargs["user_prompt"]
            assert "false_positive_corrections" not in kwargs["user_prompt"]
            return {
                "success": True,
                "tool_call": {
                    "missed_detection_corrections": [
                        {
                            "missed_index": 1,
                            "class_name": "chanche",
                            "class_name_cn": "铲车",
                            "bbox_normalized": {
                                "x_min": 0.1,
                                "y_min": 0.2,
                                "x_max": 0.4,
                                "y_max": 0.6,
                            },
                            "confidence": 0.71,
                            "description": "校验确认漏报的铲车",
                        }
                    ],
                    "summary": "只补充校验确认的漏报 bbox",
                },
            }

    detections = [
        {
            "class_id": 4,
            "class_name": "diaoche",
            "confidence": 0.89,
            "bbox": [10, 20, 110, 90],
        }
    ]
    verification = {
        "success": True,
        "data": {
            "false_positives": [
                {
                    "detection_index": 1,
                    "reported_class": "diaoche",
                    "actual_class": "wajueji",
                    "confidence": 0.84,
                    "reason": "检测框覆盖挖掘机，类别不是吊车",
                }
            ],
            "missed_detections": [
                {
                    "actual_class": "chanche",
                    "actual_class_cn": "铲车",
                    "confidence": 0.79,
                    "location": "画面左侧",
                    "description": "清晰可见，未被任何检测框覆盖",
                    "confidence_level": "high",
                }
            ],
        },
    }

    service = CorrectionService(llm_client=FakeLLM(), output_dir=tmp_path / "out")
    result = service.correct(str(image_path), detections, verification=verification)

    assert result["verification"] is verification
    assert result["correction"]["false_positive_corrections"][0]["corrected_class"] == "wajueji"
    assert result["correction"]["false_positive_corrections"][0]["confidence"] == 0.84
    assert result["correction"]["missed_detection_corrections"][0]["class_name"] == "chanche"
    assert result["corrected_detections"][0]["class_name"] == "wajueji"
    assert result["corrected_detections"][1]["bbox"] == [20, 20, 80, 60]
    assert Path(result["artifacts"]["small_model_image"]).exists()
    assert Path(result["artifacts"]["corrected_image"]).exists()

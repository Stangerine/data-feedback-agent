"""Prompt regression tests."""

from prompts import SYSTEM_PROMPT, build_verify_prompt


def test_prompt_prevents_double_counting_false_positive_as_missed_detection():
    prompt = SYSTEM_PROMPT + "\n" + build_verify_prompt(
        [
            {
                "class_name": "diaoche",
                "confidence": 0.89,
                "bbox": [437, 511, 617, 686],
            }
        ]
    )

    assert "同一个实际目标" in prompt
    assert "不能同时记录为误报和漏报" in prompt
    assert "优先记录为误报" in prompt
    assert "完全没有任何检测框覆盖" in prompt


def test_prompt_contains_vehicle_class_features_and_conservative_missed_rules():
    prompt = SYSTEM_PROMPT + "\n" + build_verify_prompt([])

    assert "挖掘机" in prompt
    assert "挖臂" in prompt
    assert "铲斗" in prompt
    assert "吊车" in prompt
    assert "起重臂" in prompt
    assert "柔性钢丝绳" in prompt
    assert "吊钩" in prompt
    assert "打桩机" in prompt
    assert "刚性桅杆" in prompt
    assert "铲车" in prompt
    assert "大容量开口铲斗" in prompt

    assert "降低漏报误判" in prompt
    assert "不确定" in prompt
    assert "不要记录为漏报" in prompt
    assert "summary" in prompt

    assert "false_positives 中每条误报必须输出 confidence" in prompt
    assert "missed_detections 中每条漏报必须输出 confidence" in prompt
    assert "0 到 1 的数字" in prompt

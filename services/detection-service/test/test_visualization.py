"""Visualization label tests."""

from services.visualization import build_detection_label


def test_corrected_false_positive_label_uses_llm_confidence_not_small_model_confidence():
    label = build_detection_label(
        1,
        {
            "class_name": "wajueji",
            "confidence": 0.95,
            "correction_confidence": 0.73,
            "original_class_name": "diaoche",
            "source": "llm_corrected",
        },
    )

    assert label == "1. wajueji 0.73 (diaoche -> wajueji)"
    assert "0.95" not in label


def test_small_model_and_llm_added_labels_keep_detection_confidence():
    assert (
        build_detection_label(
            1,
            {"class_name": "diaoche", "confidence": 0.95, "source": "small_model"},
        )
        == "1. diaoche 0.95"
    )
    assert (
        build_detection_label(
            2,
            {"class_name": "chanche", "confidence": 0.68, "source": "llm_added"},
        )
        == "2. chanche 0.68 (LLM_ADD)"
    )

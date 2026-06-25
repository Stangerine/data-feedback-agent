"""Module boundary regression tests."""

from correction import (
    BBOX_CORRECTION_SYSTEM_PROMPT,
    BBOX_CORRECTION_TOOL,
    CorrectionService,
    build_missed_bbox_prompt,
    render_detections,
)
from prompts import (
    SYSTEM_PROMPT,
    VEHICLE_CLASSES,
    VEHICLE_CLASS_FEATURES,
    VERIFICATION_TOOL,
    build_verify_prompt,
)
from prompts.correction import build_missed_bbox_prompt as package_bbox_prompt
from prompts.verification import build_verify_prompt as package_verify_prompt
from scenarios.engineering_vehicles import CLASS_IDS, SCENARIO
from services.correction_service import CorrectionService as PackageCorrectionService
from services.visualization import render_detections as package_render_detections
from tools.function_calling import BBOX_CORRECTION_TOOL as package_bbox_tool
from tools.function_calling import VERIFICATION_TOOL as package_verification_tool


def test_prompt_tools_and_scenario_are_separate_modules():
    assert SCENARIO.name == "engineering_vehicles"
    assert SCENARIO.classes is VEHICLE_CLASSES
    assert "wajueji" in CLASS_IDS
    assert "挖掘机" in VEHICLE_CLASS_FEATURES

    assert SYSTEM_PROMPT
    assert build_verify_prompt is package_verify_prompt
    assert build_missed_bbox_prompt is package_bbox_prompt

    assert VERIFICATION_TOOL is package_verification_tool
    assert BBOX_CORRECTION_TOOL is package_bbox_tool


def test_legacy_correction_imports_remain_compatible():
    assert CorrectionService is PackageCorrectionService
    assert render_detections is package_render_detections
    assert "校验结果是唯一事实来源" in BBOX_CORRECTION_SYSTEM_PROMPT

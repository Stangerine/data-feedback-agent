"""Prompt package compatibility exports."""

from scenarios.engineering_vehicles import VEHICLE_CLASSES, VEHICLE_CLASS_FEATURES
from tools.function_calling import BBOX_CORRECTION_TOOL, VERIFICATION_TOOL

from .correction import BBOX_CORRECTION_SYSTEM_PROMPT, build_missed_bbox_prompt
from .verification import SYSTEM_PROMPT, build_verify_prompt, build_verify_prompt_with_gt

__all__ = [
    "BBOX_CORRECTION_SYSTEM_PROMPT",
    "BBOX_CORRECTION_TOOL",
    "SYSTEM_PROMPT",
    "VEHICLE_CLASSES",
    "VEHICLE_CLASS_FEATURES",
    "VERIFICATION_TOOL",
    "build_missed_bbox_prompt",
    "build_verify_prompt",
    "build_verify_prompt_with_gt",
]

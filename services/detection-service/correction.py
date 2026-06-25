"""Compatibility exports for detection correction.

New code should import from ``services.correction_service``,
``services.visualization``, ``prompts.correction``, and
``tools.function_calling`` directly.
"""

from prompts.correction import BBOX_CORRECTION_SYSTEM_PROMPT, build_missed_bbox_prompt
from services.correction_service import (
    CorrectionService,
    build_corrected_detections,
    build_correction_from_verification,
    normalize_class_name,
    normalized_bbox_to_pixels,
)
from services.visualization import render_detections
from tools.function_calling import BBOX_CORRECTION_TOOL

__all__ = [
    "BBOX_CORRECTION_SYSTEM_PROMPT",
    "BBOX_CORRECTION_TOOL",
    "CorrectionService",
    "build_corrected_detections",
    "build_correction_from_verification",
    "build_missed_bbox_prompt",
    "normalize_class_name",
    "normalized_bbox_to_pixels",
    "render_detections",
]

"""Prompt formatting helpers."""

from scenarios.engineering_vehicles import VEHICLE_CLASSES


def format_detections(detections: list) -> str:
    """Format small-model detection results for LLM prompts."""
    if not detections:
        return "小模型未检测到任何目标"

    lines = []
    for i, det in enumerate(detections, 1):
        cls = det.get("class_name", "unknown")
        cn = VEHICLE_CLASSES.get(cls, cls)
        conf = det.get("confidence", 0)
        bbox = det.get("bbox", [0, 0, 0, 0])
        lines.append(
            f"  {i}. 类别: {cn}({cls}), 置信度: {conf:.2f}, "
            f"检测框: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]"
        )
    return "\n".join(lines)

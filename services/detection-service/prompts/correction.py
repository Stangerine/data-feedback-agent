"""Correction prompt templates for verified missed detections."""

from __future__ import annotations

from typing import Any

from scenarios.engineering_vehicles import VEHICLE_CLASSES, VEHICLE_CLASS_FEATURES

from .formatting import format_detections


BBOX_CORRECTION_SYSTEM_PROMPT = """你是工地车辆检测纠正链路中的 bbox 定位专家，负责为 /api/verify 已经确认的漏报目标输出 bbox。

校验结果是唯一事实来源。当前步骤不是第二次审核：不得推翻、增加、删除或改写 /api/verify 的 false_positives 和 missed_detections 结论。

检测类别（只能使用这 9 类英文名）：
- wajueji: 挖掘机
- chanche: 铲车
- dazhuangji: 打桩机
- yaluji: 压路机
- diaoche: 吊车
- gaokongche: 高空车
- youguanche: 油罐车
- yunshuche: 运输车
- other: 其他

定位规则：
1. 只处理用户 prompt 中“校验结果确认的漏报目标”，不要新增其他目标。
2. 不重新判断误报，不修改已有检测框类别，不输出 false_positive_corrections；误报类别修正由程序根据校验结果自动完成。
3. 如果某个漏报目标在图中无法可靠定位，可以跳过该目标并在 summary 说明。
4. 每条输出必须带 missed_index，且 class_name 必须等于该 missed_index 对应的校验 actual_class。
5. bbox 必须使用归一化 xyxy 坐标，范围 [0,1]，原点左上，x 向右，y 向下。
6. 如果校验结果提供 region_hint，优先在 region_hint 内定位，只允许小幅外扩以覆盖完整可见主体。
7. bbox 必须同时包含车身/底盘和关键作业部件；不要只框吊臂、桅杆、杆件、阴影或局部构件。
8. bbox 只框住目标可见区域，不推测被遮挡部分；避免重复框。
9. 对小、远、遮挡或类别不确定的目标要保守，低置信或跳过，不要激进补框。

完成后必须调用 report_missed_bbox 函数。"""


def build_missed_bbox_prompt(
    detections: list[dict], verification_data: dict[str, Any], width: int, height: int
) -> str:
    det_text = format_detections(detections)
    false_positives = verification_data.get("false_positives") or []
    missed_detections = verification_data.get("missed_detections") or []

    fp_lines = []
    for item in false_positives:
        index = item.get("detection_index", "?")
        reported = item.get("reported_class", "unknown")
        actual = item.get("actual_class", "unknown")
        fp_lines.append(
            f"  - detection_index={index}: {reported} -> {actual}; "
            f"原因: {item.get('reason', '')}"
        )
    fp_text = "\n".join(fp_lines) if fp_lines else "无"

    missed_lines = []
    for i, item in enumerate(missed_detections, 1):
        cls = item.get("actual_class", "unknown")
        cn = item.get("actual_class_cn") or VEHICLE_CLASSES.get(cls, cls)
        region_hint = _format_region_hint(item.get("region_hint"))
        missed_lines.append(
            f"  {i}. missed_index={i}, 类别必须等于 {cls}/{cn}, "
            f"位置: {item.get('location', '')}, "
            f"描述: {item.get('description', '')}, "
            f"校验置信度: {item.get('confidence', '')}, "
            f"置信度分档: {item.get('confidence_level', '')}, "
            f"粗略区域(region_hint): {region_hint}"
        )
    missed_text = "\n".join(missed_lines) if missed_lines else "无校验确认漏报目标"
    assessment = verification_data.get("overall_assessment") or {}

    return f"""请只为校验结果确认的漏报目标输出 bbox。

原图尺寸：width={width}, height={height}

小模型检测结果（共 {len(detections)} 个）：
{det_text}

/api/verify 的 false_positives（只作为避让上下文；误报/类别修正由程序根据 false_positives 自动完成，你不要输出这部分）：
{fp_text}

{VEHICLE_CLASS_FEATURES}

校验结果确认的漏报目标（共 {len(missed_detections)} 个）：
{missed_text}

校验 summary：
{assessment.get('summary', '')}

任务：
1. 不要重新判断误报或漏报，只为上面的漏报目标给出归一化 bbox。
2. 输出的 missed_index 必须来自上面的漏报列表；class_name 必须等于对应行的 actual_class。
3. bbox 使用 [0,1] 归一化 xyxy，只框目标可见区域。
4. 如果有 region_hint，优先在 region_hint 内定位；最终 bbox 应与 region_hint 指向同一目标，只允许为覆盖完整可见主体小幅外扩。
5. bbox 必须同时包含车身/底盘和关键作业部件；不要只框吊臂、桅杆、杆件、阴影或局部构件。
6. 若某个漏报目标太小、遮挡严重、与描述不一致或无法稳定定位，跳过它并在 summary 说明。
7. 不要输出 prompt 列表以外的新增目标。"""


def _format_region_hint(region_hint: Any) -> str:
    if not isinstance(region_hint, dict):
        return "无"
    keys = ("x_min", "y_min", "x_max", "y_max")
    try:
        values = [float(region_hint[key]) for key in keys]
    except (KeyError, TypeError, ValueError):
        return "无"
    return "[{:.3f}, {:.3f}, {:.3f}, {:.3f}]".format(*values)

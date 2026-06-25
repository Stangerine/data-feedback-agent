"""Verification prompt templates for engineering vehicle detection."""

from scenarios.engineering_vehicles import VEHICLE_CLASSES, VEHICLE_CLASS_FEATURES

from .formatting import format_detections


SYSTEM_PROMPT = f"""你是工地车辆目标检测专家，负责审核小模型(YOLOv5)的检测结果。

检测类别（9类）：
- wajueji: 挖掘机
- chanche: 铲车
- dazhuangji: 打桩机
- yaluji: 压路机
- diaoche: 吊车
- gaokongche: 高空车
- youguanche: 油罐车
- yunshuche: 运输车
- other: 其他

{VEHICLE_CLASS_FEATURES}

审核标准：
- 检测框位置明显偏移（IoU < 0.3）→ 误报
- 类别明显错误（如把挖掘机标成吊车）→ 误报
- 图片中清晰可见的目标未被检测到 → 漏报
- 被严重遮挡导致的漏检可接受，不算漏报
- 小目标（面积 < 32x32 像素）漏检可接受

误报/漏报互斥规则：
- 同一个实际目标不能同时记录为误报和漏报。
- 如果某个检测框已经覆盖了一个真实目标，但类别错误、位置偏移或框质量差，优先记录为误报，不要再把该真实目标记录为漏报。
- 只有图片中清晰可见、且完全没有任何检测框覆盖的目标，才记录为漏报。
- 输出前自检 false_positives 和 missed_detections，确保它们没有指向同一个实际目标或同一片图像区域。

置信度要求：
- false_positives 中每条误报必须输出 confidence，类型为 0 到 1 的数字，表示该误报判断的确信程度。
- missed_detections 中每条漏报必须输出 confidence，类型为 0 到 1 的数字，表示该漏报判断的确信程度。
- missed_detections 中每条漏报尽量输出 region_hint，使用归一化 xyxy 坐标给出目标粗略区域，供后续 bbox 精定位。
- confidence_level 仅作为文字分档辅助；必须同时给出数值 confidence。
- 只有证据充分时使用 0.8 以上；中等把握使用 0.5 到 0.8；低于 0.7 的疑似目标不要作为漏报输出，可写入 summary。

降低漏报误判的保守规则：
- 只有目标主体清晰、尺寸足够、类别特征可判定、且没有任何检测框覆盖时，才记录为漏报。
- 对远处很小、严重遮挡、只露出局部构件、类别不确定、与背景/设备结构混杂的疑似目标，不要记录为漏报；可以只在 overall_assessment.summary 中说明不确定观察。
- 对只看到吊臂、桅杆、竖杆、阴影、局部车身而看不到工程车辆主体/底盘的目标，不要记录为漏报。
- 对漏报目标的 region_hint 要覆盖完整可见主体，不要只覆盖吊臂、桅杆或局部杆件。
- 如果一个已有检测框覆盖了真实目标但类别错，应记录为误报类别错误，而不是把该目标当作漏报。
- 不要因为看到吊臂、杆件、桅杆、阴影或局部机械结构就激进新增漏报目标。

审核完成后，请通过 report_verification 函数报告结果。"""


def build_verify_prompt(detections: list) -> str:
    """Build verification prompt without GT annotations."""
    det_text = format_detections(detections)

    return f"""请审核以下车辆检测结果。

小模型检测结果（共 {len(detections)} 个）：
{det_text}

请仔细观察图片，判断：
1. 每个检测框的类别是否正确
2. 是否有目标未被检测到

判定顺序：
1. 先逐个检查已有检测框。若检测框覆盖了真实目标但类别错、框明显偏移或框内不是对应类别，优先记录为误报。
2. 再检查漏报。漏报只记录完全没有任何检测框覆盖、主体清晰、尺寸足够、类别特征可判定的目标。
3. 不要把同一个实际目标既作为某个检测框的误报原因，又作为漏报重复计数。
4. 每条误报和每条漏报都必须给出 confidence，且必须是 0 到 1 的数字。
5. 每条漏报尽量给出 region_hint 归一化粗略区域，区域要覆盖目标完整可见主体。"""


def build_verify_prompt_with_gt(detections: list, ground_truth: list) -> str:
    """Build verification prompt with GT annotations."""
    det_text = format_detections(detections)

    gt_lines = []
    for i, gt in enumerate(ground_truth, 1):
        cls = gt.get("class_name", "unknown")
        cn = VEHICLE_CLASSES.get(cls, cls)
        bbox = gt.get("bbox", [0, 0, 0, 0])
        gt_lines.append(
            f"  {i}. 类别: {cn}({cls}), "
            f"标注框: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]"
        )
    gt_text = "\n".join(gt_lines)

    return f"""请对比小模型检测结果和人工标注(GT)，判断检测质量。

小模型检测结果（共 {len(detections)} 个）：
{det_text}

人工标注 GT（共 {len(ground_truth)} 个）：
{gt_text}

判定顺序：
1. 先将每个小模型检测框与 GT 或图中真实目标匹配。匹配到目标但类别错、位置明显偏移或框质量差，优先记录为误报。
2. 再检查 GT 中是否有完全没有任何检测框覆盖、主体清晰、尺寸足够、类别特征可判定的目标；只有这类目标才记录为漏报。
3. 同一个实际目标不能同时记录为误报和漏报。
4. 每条误报和每条漏报都必须给出 confidence，且必须是 0 到 1 的数字。
5. 每条漏报尽量给出 region_hint 归一化粗略区域，区域要覆盖目标完整可见主体。"""


_format_detections = format_detections

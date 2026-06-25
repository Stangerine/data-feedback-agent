"""Detection result visualization helpers."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def build_detection_label(index: int, detection: dict) -> str:
    """Build the text label shown above a rendered detection box."""
    source = detection.get("source", "small_model")
    label = f"{index}. {detection.get('class_name', 'unknown')}"

    confidence = detection.get("confidence")
    if source == "llm_corrected":
        confidence = detection.get("correction_confidence", confidence)
    if confidence is not None:
        label += f" {float(confidence):.2f}"

    if source == "llm_corrected":
        label += (
            f" ({detection.get('original_class_name')} -> "
            f"{detection.get('class_name')})"
        )
    elif source == "llm_added":
        label += " (LLM_ADD)"

    return label


def render_detections(
    image_path: str, detections: list[dict], output_path: str, title: str
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    palette = {
        "small_model": (0, 128, 255),
        "llm_corrected": (255, 140, 0),
        "llm_added": (220, 0, 0),
    }
    draw.text((8, 8), title, fill=(255, 255, 255), font=font)

    for idx, det in enumerate(detections, 1):
        bbox = [float(v) for v in det.get("bbox", [0, 0, 0, 0])]
        source = det.get("source", "small_model")
        color = palette.get(source, (0, 255, 0))
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)

        label = build_detection_label(idx, det)

        text_bbox = draw.textbbox((x1, max(0, y1 - 14)), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, max(0, y1 - 14)), label, fill=(255, 255, 255), font=font)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)

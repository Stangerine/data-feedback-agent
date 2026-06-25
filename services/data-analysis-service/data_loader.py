"""数据加载器 — 解析 YOLO/VOC 标注，构建数据集画像"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from PIL import Image

from schemas import (
    BBox,
    BBoxStats,
    ClassDistribution,
    ImageAnnotation,
)

# 类别映射
CLASS_NAMES = {
    0: "wajueji", 1: "chanche", 2: "dazhuangji", 3: "yaluji",
    4: "diaoche", 5: "gaokongche", 6: "youguanche", 7: "yunshuche",
    8: "other",
}
CLASS_NAME_TO_ID = {v: k for k, v in CLASS_NAMES.items()}


def parse_yolo_label(label_path: str) -> List[BBox]:
    """解析 YOLO 格式标注文件"""
    bboxes = []
    if not os.path.exists(label_path):
        return bboxes
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                class_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                conf = float(parts[5]) if len(parts) > 5 else 1.0
                bboxes.append(BBox(
                    class_id=class_id,
                    class_name=CLASS_NAMES.get(class_id, f"class_{class_id}"),
                    cx=cx, cy=cy, w=w, h=h,
                    confidence=conf,
                ))
            except (ValueError, IndexError):
                continue
    return bboxes


def parse_voc_xml(xml_path: str) -> Tuple[List[BBox], int, int]:
    """解析 Pascal VOC 格式标注文件，返回 (bboxes, width, height)"""
    bboxes = []
    width, height = 0, 0
    if not os.path.exists(xml_path):
        return bboxes, width, height
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find("size")
        if size is not None:
            width = int(size.find("width").text or 0)
            height = int(size.find("height").text or 0)
        for obj in root.findall("object"):
            name_elem = obj.find("name")
            if name_elem is None:
                continue
            class_name = name_elem.text.strip()
            class_id = CLASS_NAME_TO_ID.get(class_name, -1)
            bbox_elem = obj.find("bndbox")
            if bbox_elem is None:
                continue
            xmin = float(bbox_elem.find("xmin").text or 0)
            ymin = float(bbox_elem.find("ymin").text or 0)
            xmax = float(bbox_elem.find("xmax").text or 0)
            ymax = float(bbox_elem.find("ymax").text or 0)
            # 转换为 YOLO 格式
            cx = (xmin + xmax) / 2 / width if width > 0 else 0
            cy = (ymin + ymax) / 2 / height if height > 0 else 0
            w = (xmax - xmin) / width if width > 0 else 0
            h = (ymax - ymin) / height if height > 0 else 0
            bboxes.append(BBox(
                class_id=class_id,
                class_name=class_name,
                cx=cx, cy=cy, w=w, h=h,
                xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax,
            ))
    except (ET.ParseError, AttributeError):
        pass
    return bboxes, width, height


def load_annotations(data_dir: str) -> List[ImageAnnotation]:
    """加载数据集的所有标注"""
    images_dir = os.path.join(data_dir, "images")
    labels_dir = os.path.join(data_dir, "labels")
    xml_dir = os.path.join(data_dir, "xml")

    if not os.path.isdir(images_dir):
        # 尝试直接从 data_dir 加载 (扁平目录)
        images_dir = data_dir

    annotations = []
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    # 收集所有图片文件
    image_files = sorted([
        f for f in os.listdir(images_dir)
        if os.path.splitext(f)[1].lower() in image_extensions
    ])

    for fname in image_files:
        image_path = os.path.join(images_dir, fname)
        base_name = os.path.splitext(fname)[0]

        # 优先使用 YOLO 标注
        label_path = os.path.join(labels_dir, f"{base_name}.txt")
        xml_path = os.path.join(xml_dir, f"{base_name}.xml")

        bboxes = []
        width, height = 0, 0

        if os.path.exists(label_path):
            bboxes = parse_yolo_label(label_path)
        elif os.path.exists(xml_path):
            bboxes, width, height = parse_voc_xml(xml_path)

        # 如果没有标注文件，尝试从 XML 获取图片尺寸
        if width == 0 and height == 0 and os.path.exists(xml_path):
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                size = root.find("size")
                if size is not None:
                    width = int(size.find("width").text or 0)
                    height = int(size.find("height").text or 0)
            except (ET.ParseError, AttributeError):
                pass

        # 从图片获取尺寸
        if width == 0 or height == 0:
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
            except Exception:
                pass

        annotations.append(ImageAnnotation(
            filename=fname,
            image_path=image_path,
            width=width,
            height=height,
            bboxes=bboxes,
        ))

    return annotations


def compute_class_distribution(annotations: List[ImageAnnotation]) -> List[ClassDistribution]:
    """计算类别分布"""
    counter: Counter = Counter()
    for ann in annotations:
        for bbox in ann.bboxes:
            counter[bbox.class_id] += 1

    total = sum(counter.values())
    if total == 0:
        return []

    distribution = []
    for class_id in sorted(counter.keys()):
        count = counter[class_id]
        distribution.append(ClassDistribution(
            class_id=class_id,
            class_name=CLASS_NAMES.get(class_id, f"class_{class_id}"),
            count=count,
            percentage=round(count / total * 100, 2),
        ))
    return distribution


def compute_bbox_stats(annotations: List[ImageAnnotation]) -> BBoxStats:
    """计算边界框统计信息"""
    all_bboxes: List[BBox] = []
    for ann in annotations:
        all_bboxes.extend(ann.bboxes)

    if not all_bboxes:
        return BBoxStats(total_count=0, mean_area_ratio=0, mean_aspect_ratio=0)

    area_ratios = []
    aspect_ratios = []
    size_counter = Counter()

    for bbox, ann in [(b, a) for a in annotations for b in a.bboxes]:
        if ann.width > 0 and ann.height > 0:
            area = bbox.w * bbox.h  # 归一化面积
            area_ratios.append(area)
            aspect_ratios.append(bbox.w / bbox.h if bbox.h > 0 else 0)
            # 按面积分桶: small < 0.5%, 0.5% <= medium < 2%, large >= 2%
            area_pct = area * 100
            if area_pct < 0.5:
                size_counter["small"] += 1
            elif area_pct < 2.0:
                size_counter["medium"] += 1
            else:
                size_counter["large"] += 1

    return BBoxStats(
        total_count=len(all_bboxes),
        mean_area_ratio=round(sum(area_ratios) / len(area_ratios), 6) if area_ratios else 0,
        mean_aspect_ratio=round(sum(aspect_ratios) / len(aspect_ratios), 4) if aspect_ratios else 0,
        size_distribution=dict(size_counter),
    )


def compute_image_dims(annotations: List[ImageAnnotation]) -> Dict[str, int]:
    """统计图片尺寸分布"""
    dims_counter: Counter = Counter()
    for ann in annotations:
        key = f"{ann.width}x{ann.height}"
        dims_counter[key] += 1
    return dict(dims_counter.most_common(20))


def profile_dataset(data_dir: str) -> dict:
    """生成数据集画像的完整统计"""
    annotations = load_annotations(data_dir)
    class_dist = compute_class_distribution(annotations)
    bbox_stats = compute_bbox_stats(annotations)
    image_dims = compute_image_dims(annotations)

    return {
        "data_dir": data_dir,
        "image_count": len(annotations),
        "total_objects": bbox_stats.total_count,
        "class_distribution": [cd.model_dump() for cd in class_dist],
        "bbox_stats": bbox_stats.model_dump(),
        "image_dims": image_dims,
        "annotations": annotations,  # 保留原始标注供后续分析
    }

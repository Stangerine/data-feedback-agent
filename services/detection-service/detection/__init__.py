"""
检测客户端 — 调用线上 YOLO 检测 API
"""

import base64
from pathlib import Path

import requests


class DetectionClient:
    """调用线上 YOLOv5 检测 API，返回统一格式的检测结果"""

    def __init__(self, api_url: str, model_id: str, box_threshold: float = 0.5,
                 timeout: int = 60):
        self.api_url = api_url
        self.model_id = model_id
        self.box_threshold = box_threshold
        self.timeout = timeout
        print(f"[DetectionClient] url={api_url}, model={model_id}, threshold={box_threshold}")

    def detect(self, image_path: str, threshold: float | None = None) -> list[dict]:
        """
        对单张图片执行检测

        Returns:
            list[dict]: [{ class_id, class_name, confidence, bbox: [x1,y1,x2,y2] }]
        """
        threshold = threshold if threshold is not None else self.box_threshold

        # 读取图片并编码
        suffix = Path(image_path).suffix.lower().lstrip(".")
        mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "bmp": "bmp"}.get(suffix, "jpeg")

        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "modelId": self.model_id,
            "version": "v0.1",
            "imageValue": f"data:image/{mime};base64,{img_b64}",
            "imageType": "base64",
            "boxThreshold": str(threshold),
        }

        resp = requests.post(self.api_url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        result = resp.json()

        if result.get("code") != 200:
            raise RuntimeError(f"检测 API 错误: {result.get('msg', '未知错误')}")

        return self._parse(result)

    def _parse(self, result: dict) -> list[dict]:
        """解析 API 响应，转为统一格式"""
        objects = (result
                   .get("data", {})
                   .get("detection_results", {})
                   .get("objects", []))

        detections = []
        for obj in objects:
            conf = float(obj.get("confidence", 0))
            if conf < self.box_threshold:
                continue

            bbox_raw = obj.get("bounding_box", {})
            x = float(bbox_raw.get("x", 0))
            y = float(bbox_raw.get("y", 0))
            w = float(bbox_raw.get("width", 0))
            h = float(bbox_raw.get("height", 0))

            detections.append({
                "class_id": obj.get("class_id", -1),
                "class_name": obj.get("class_name", "unknown"),
                "confidence": conf,
                "bbox": [x, y, x + w, y + h],
            })

        return detections

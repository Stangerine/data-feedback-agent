"""Function calling schemas for verification and correction."""

from scenarios.engineering_vehicles import VEHICLE_CLASSES


VERIFICATION_TOOL = {
    "type": "function",
    "function": {
        "name": "report_verification",
        "description": "报告车辆检测校验结果。审核完成后必须调用此函数。",
        "parameters": {
            "type": "object",
            "properties": {
                "false_positives": {
                    "type": "array",
                    "description": "误报列表",
                    "items": {
                        "type": "object",
                        "properties": {
                            "detection_index": {
                                "type": "integer",
                                "description": "检测结果序号(从1开始)",
                            },
                            "reported_class": {
                                "type": "string",
                                "description": "小模型给出的类别",
                            },
                            "actual_class": {
                                "type": "string",
                                "description": "实际类别",
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                                "description": "误报判断置信度，0到1",
                            },
                            "reason": {
                                "type": "string",
                                "description": "误报原因",
                            },
                        },
                        "required": [
                            "detection_index",
                            "reported_class",
                            "actual_class",
                            "confidence",
                            "reason",
                        ],
                    },
                },
                "missed_detections": {
                    "type": "array",
                    "description": "漏报列表",
                    "items": {
                        "type": "object",
                        "properties": {
                            "actual_class": {
                                "type": "string",
                                "description": "实际类别英文名",
                            },
                            "actual_class_cn": {
                                "type": "string",
                                "description": "实际类别中文名",
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                                "description": "漏报判断置信度，0到1",
                            },
                            "region_hint": {
                                "type": "object",
                                "description": (
                                    "漏报目标粗略区域，归一化 xyxy 坐标，"
                                    "用于后续 bbox 精定位；不确定时可省略"
                                ),
                                "properties": {
                                    "x_min": {
                                        "type": "number",
                                        "minimum": 0,
                                        "maximum": 1,
                                    },
                                    "y_min": {
                                        "type": "number",
                                        "minimum": 0,
                                        "maximum": 1,
                                    },
                                    "x_max": {
                                        "type": "number",
                                        "minimum": 0,
                                        "maximum": 1,
                                    },
                                    "y_max": {
                                        "type": "number",
                                        "minimum": 0,
                                        "maximum": 1,
                                    },
                                },
                                "required": ["x_min", "y_min", "x_max", "y_max"],
                            },
                            "location": {
                                "type": "string",
                                "description": "在图片中的位置",
                            },
                            "description": {
                                "type": "string",
                                "description": "描述",
                            },
                            "confidence_level": {
                                "type": "string",
                                "enum": ["high", "medium", "low"],
                                "description": "确信度",
                            },
                        },
                        "required": ["actual_class", "confidence"],
                    },
                },
                "overall_assessment": {
                    "type": "object",
                    "description": "总体评估",
                    "properties": {
                        "total_detections": {"type": "integer"},
                        "false_positive_count": {"type": "integer"},
                        "missed_detection_count": {"type": "integer"},
                        "detection_quality": {
                            "type": "string",
                            "enum": ["good", "fair", "poor"],
                        },
                        "summary": {"type": "string"},
                    },
                },
            },
            "required": [
                "false_positives",
                "missed_detections",
                "overall_assessment",
            ],
        },
    },
}


BBOX_CORRECTION_TOOL = {
    "type": "function",
    "function": {
        "name": "report_missed_bbox",
        "description": "报告已确认漏报目标的 bbox。",
        "parameters": {
            "type": "object",
            "properties": {
                "missed_detection_corrections": {
                    "type": "array",
                    "description": "校验确认漏报目标的 bbox 列表",
                    "items": {
                        "type": "object",
                        "properties": {
                            "missed_index": {
                                "type": "integer",
                                "description": "对应校验漏报列表序号，从1开始",
                            },
                            "class_name": {
                                "type": "string",
                                "enum": list(VEHICLE_CLASSES.keys()),
                                "description": "漏报目标类别英文名",
                            },
                            "class_name_cn": {
                                "type": "string",
                                "description": "漏报目标类别中文名",
                            },
                            "bbox_normalized": {
                                "type": "object",
                                "description": "归一化 xyxy bbox，范围 [0,1]",
                                "properties": {
                                    "x_min": {"type": "number"},
                                    "y_min": {"type": "number"},
                                    "x_max": {"type": "number"},
                                    "y_max": {"type": "number"},
                                },
                                "required": ["x_min", "y_min", "x_max", "y_max"],
                            },
                            "confidence": {
                                "type": "number",
                                "description": "新增目标置信度 0-1",
                            },
                            "description": {
                                "type": "string",
                                "description": "目标位置和外观描述",
                            },
                        },
                        "required": [
                            "missed_index",
                            "class_name",
                            "bbox_normalized",
                            "confidence",
                            "description",
                        ],
                    },
                },
                "summary": {
                    "type": "string",
                    "description": "纠正总结",
                },
            },
            "required": [
                "missed_detection_corrections",
                "summary",
            ],
        },
    },
}

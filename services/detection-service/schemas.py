"""
数据模型定义 — Pydantic schema
"""

from typing import Optional
from pydantic import BaseModel, Field, model_validator


# ── 检测结果 ─────────────────────────────────────────────────

class Detection(BaseModel):
    """单个检测框"""
    class_id: int = Field(..., description="类别 ID")
    class_name: str = Field(..., description="类别英文名")
    class_name_cn: str = Field("", description="类别中文名")
    confidence: float = Field(..., ge=0, le=1, description="置信度")
    bbox: list[float] = Field(..., description="[x1, y1, x2, y2]")


class GroundTruth(BaseModel):
    """人工标注"""
    class_name: str
    bbox: list[float]


# ── 大模型结构化输出 ─────────────────────────────────────────

class NormalizedBBox(BaseModel):
    """归一化 bbox，xyxy，范围 0 到 1"""
    x_min: float = Field(..., ge=0, le=1)
    y_min: float = Field(..., ge=0, le=1)
    x_max: float = Field(..., ge=0, le=1)
    y_max: float = Field(..., ge=0, le=1)

    @model_validator(mode="after")
    def validate_order(self):
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("bbox must satisfy x_max>x_min and y_max>y_min")
        return self

class FalsePositive(BaseModel):
    """误报"""
    detection_index: int = Field(..., description="对应检测结果序号(从1开始)")
    reported_class: str = Field(..., description="小模型给出的类别")
    actual_class: str = Field(..., description="实际类别")
    confidence: float = Field(..., ge=0, le=1, description="误报判断置信度，0到1")
    reason: str = Field(..., description="误报原因")
    bbox: list[float] = Field(default=[], description="检测框坐标")


class MissedDetection(BaseModel):
    """漏报"""
    actual_class: str = Field(..., description="实际类别英文名")
    actual_class_cn: str = Field("", description="实际类别中文名")
    confidence: float = Field(..., ge=0, le=1, description="漏报判断置信度，0到1")
    region_hint: Optional[NormalizedBBox] = Field(
        None, description="漏报目标粗略区域，归一化 xyxy，用于后续 bbox 精定位"
    )
    location: str = Field("", description="在图片中的位置")
    description: str = Field("", description="描述")
    confidence_level: str = Field("medium", description="确信度 high/medium/low")


class OverallAssessment(BaseModel):
    """总体评估"""
    total_detections: int = 0
    false_positive_count: int = 0
    missed_detection_count: int = 0
    detection_quality: str = Field("good", description="good/fair/poor")
    summary: str = ""


class VerificationResult(BaseModel):
    """大模型校验结果"""
    false_positives: list[FalsePositive] = []
    missed_detections: list[MissedDetection] = []
    overall_assessment: OverallAssessment = Field(default_factory=OverallAssessment)


# ── API 请求/响应 ────────────────────────────────────────────

class VerifyRequest(BaseModel):
    """单图校验请求"""
    image_path: str = Field(..., description="图片路径")
    box_threshold: Optional[float] = Field(None, description="置信度阈值(覆盖默认)")
    ground_truth: Optional[list[GroundTruth]] = Field(None, description="人工标注(可选)")


class VerifyDirectRequest(BaseModel):
    """直接校验请求(跳过检测API)"""
    image_path: str = Field(..., description="图片路径")
    detections: list[Detection] = Field(..., description="检测结果列表")
    ground_truth: Optional[list[GroundTruth]] = Field(None, description="人工标注(可选)")


class VerifyBatchRequest(BaseModel):
    """批量校验请求"""
    image_dir: str = Field(..., description="图片目录")
    limit: Optional[int] = Field(None, description="限制数量")
    box_threshold: Optional[float] = Field(None, description="置信度阈值")


class CorrectRequest(BaseModel):
    """单图纠正请求"""
    image_path: str = Field(..., description="图片路径")
    box_threshold: Optional[float] = Field(None, description="置信度阈值(覆盖默认)")

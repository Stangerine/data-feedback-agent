"""Pydantic 数据模型 — 数据分析服务"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── 标注数据 ──────────────────────────────────────────────

class BBox(BaseModel):
    """单个边界框"""
    class_id: int
    class_name: str = ""
    cx: float = 0.0
    cy: float = 0.0
    w: float = 0.0
    h: float = 0.0
    xmin: float = 0.0
    ymin: float = 0.0
    xmax: float = 0.0
    ymax: float = 0.0
    confidence: float = 1.0


class ImageAnnotation(BaseModel):
    """单张图片的标注信息"""
    filename: str
    image_path: str
    width: int = 0
    height: int = 0
    bboxes: List[BBox] = Field(default_factory=list)


# ── 数据集画像 ────────────────────────────────────────────

class ClassDistribution(BaseModel):
    """类别分布"""
    class_id: int
    class_name: str
    count: int
    percentage: float


class BBoxStats(BaseModel):
    """边界框统计"""
    total_count: int
    mean_area_ratio: float
    mean_aspect_ratio: float
    size_distribution: Dict[str, int] = Field(default_factory=dict)


class DatasetProfile(BaseModel):
    """数据集画像"""
    data_dir: str
    image_count: int
    total_objects: int
    class_distribution: List[ClassDistribution]
    bbox_stats: BBoxStats
    image_dims: Dict[str, int] = Field(default_factory=dict)


# ── 语义维度分析结果 ──────────────────────────────────────

class SemanticDimensionResult(BaseModel):
    """单个语义维度的分析结果"""
    dimension: str
    best_category: str
    confidence: float = Field(ge=0, le=1)
    similarities: Dict[str, float] = Field(default_factory=dict)


class DatasetSemanticDistribution(BaseModel):
    """数据集的语义维度分布统计"""
    dimension: str
    category_counts: Dict[str, int] = Field(default_factory=dict)
    category_percentages: Dict[str, float] = Field(default_factory=dict)


# ── LLM 归因分析结果 ──────────────────────────────────────

class AttributionType(str, Enum):
    """归因类型 — 与分析的7个维度对应"""
    LIGHTING = "光照问题"
    VIEWPOINT = "视角问题"
    BLUR = "清晰度问题"
    WEATHER = "天气问题"
    TIME_OF_DAY = "时段问题"
    ENVIRONMENT = "环境问题"
    CLASS_BIAS = "类别偏差"
    CLASS_CONFUSION = "类间混淆"
    OTHER = "其他"


class DimensionAttribution(BaseModel):
    """单个维度的归因分析"""
    dimension: str  # 维度名称
    category: str  # 该图片在该维度的分类
    train_coverage: float  # 训练集中该类别的占比
    is_gap: bool  # 是否是训练集的覆盖缺口
    contribution: str  # 该维度对误报/漏报的贡献描述


class LLMAttribution(BaseModel):
    """LLM 归因分析结果"""
    filename: str
    attribution_type: AttributionType
    confidence: float = Field(ge=0, le=1)
    reasoning: str  # 详细分析原因
    dimension_attributions: List[DimensionAttribution] = Field(default_factory=list)
    main_cause_dimension: str  # 主要原因维度
    feedback_suggestion: str  # 回流建议
    should_feedback: bool  # 是否建议回流


# ── API 请求/响应 ─────────────────────────────────────────

class ProfileRequest(BaseModel):
    data_dir: str


class CompareRequest(BaseModel):
    training_dir: Optional[str] = None
    test_dir: Optional[str] = None


class ScoreRequest(BaseModel):
    image_path: str
    detections: List[Dict[str, Any]] = Field(default_factory=list)
    annotation: Optional[Dict[str, Any]] = None


class ExportRequest(BaseModel):
    format: str = "json"
    min_level: str = "high"  # high / medium / low / all

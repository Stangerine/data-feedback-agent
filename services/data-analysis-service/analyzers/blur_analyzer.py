"""模糊维度分析器"""

from __future__ import annotations

from .semantic_analyzer import BaseSemanticAnalyzer


class BlurAnalyzer(BaseSemanticAnalyzer):
    """模糊分析器: sharp / motion-blur / out-of-focus"""

    DIMENSION_NAME = "blur"
    CATEGORIES = {
        "sharp": "a sharp image with clear object edges and readable details",
        "motion-blur": "an image with directional motion blur caused by movement",
        "out-of-focus": "an image where the object is soft and out of focus",
    }

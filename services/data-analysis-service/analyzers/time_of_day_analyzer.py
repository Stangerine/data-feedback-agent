"""时间维度分析器"""

from __future__ import annotations

from .semantic_analyzer import BaseSemanticAnalyzer


class TimeOfDayAnalyzer(BaseSemanticAnalyzer):
    """时间分析器: day / dusk / night"""

    DIMENSION_NAME = "timeOfDay"
    CATEGORIES = {
        "day": "a daytime image",
        "dusk": "an image captured at dusk",
        "night": "a nighttime image",
    }

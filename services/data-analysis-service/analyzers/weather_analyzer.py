"""天气维度分析器"""

from __future__ import annotations

from .semantic_analyzer import BaseSemanticAnalyzer


class WeatherAnalyzer(BaseSemanticAnalyzer):
    """天气分析器: clear / cloudy / rain / snow / fog"""

    DIMENSION_NAME = "weather"
    CATEGORIES = {
        "clear": "an outdoor image in clear sunny weather with good visibility",
        "cloudy": "an outdoor image in cloudy or overcast weather with gray clouds",
        "rain": "an outdoor image in rainy wet weather with water, rain streaks, or wet road",
        "snow": "an outdoor image with snow or icy ground",
        "fog": "an outdoor image with fog, mist, haze, or low visibility",
    }

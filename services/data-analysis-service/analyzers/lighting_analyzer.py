"""光照维度分析器"""

from __future__ import annotations

from .semantic_analyzer import BaseSemanticAnalyzer


class LightingAnalyzer(BaseSemanticAnalyzer):
    """光照分析器: bright / moderate / dim"""

    DIMENSION_NAME = "lighting"
    CATEGORIES = {
        "bright": "a bright well lit image with strong overall illumination and clearly visible details",
        "moderate": "an image with normal balanced daylight or indoor lighting, neither very bright nor very dark",
        "dim": "a dark low light image with weak illumination, night lighting, or heavy shadows",
    }

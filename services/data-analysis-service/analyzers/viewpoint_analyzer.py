"""视角维度分析器"""

from __future__ import annotations

from .semantic_analyzer import BaseSemanticAnalyzer


class ViewpointAnalyzer(BaseSemanticAnalyzer):
    """视角分析器: front / side / rear / overhead"""

    DIMENSION_NAME = "viewpoint"
    CATEGORIES = {
        "front": "a vehicle or construction machine seen from the front",
        "side": "a vehicle or construction machine seen from the left or right side",
        "rear": "a vehicle or construction machine seen from behind",
        "overhead": "a vehicle or construction machine seen from above or from an aerial top view",
    }

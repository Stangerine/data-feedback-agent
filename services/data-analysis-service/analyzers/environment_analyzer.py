"""环境维度分析器"""

from __future__ import annotations

from .semantic_analyzer import BaseSemanticAnalyzer


class EnvironmentAnalyzer(BaseSemanticAnalyzer):
    """环境分析器: indoor / urban-street / construction-site / rural-field / aerial-scene"""

    DIMENSION_NAME = "environment"
    CATEGORIES = {
        "indoor": "an indoor warehouse garage factory or workshop environment",
        "urban-street": "an urban street or city road environment with buildings or traffic",
        "construction-site": "a construction site or work zone with machinery dirt materials or barriers",
        "rural-field": "a rural field farm open land or countryside environment",
        "aerial-scene": "an aerial drone scene or high altitude overhead environment",
    }

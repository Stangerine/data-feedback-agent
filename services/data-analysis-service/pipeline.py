"""分析流水线 — 预计算训练集分布 + 单图归因分析"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from data_loader import load_annotations, compute_class_distribution
from schemas import (
    DatasetSemanticDistribution,
    LLMAttribution,
    SemanticDimensionResult,
)

logger = logging.getLogger(__name__)


class AnalysisPipeline:
    """数据分析流水线

    流程:
    1. 预计算训练集分布 (服务启动时)
    2. 单图归因分析 (每次请求时)
    """

    def __init__(self, config: dict):
        self.config = config
        self.output_dir = config.get("output", {}).get("dir", "./analysis_results")
        os.makedirs(self.output_dir, exist_ok=True)

        # 预计算的训练集分布
        self.train_class_analysis: Optional[Dict[str, Any]] = None
        self.train_semantic_distribution: Optional[Dict[str, DatasetSemanticDistribution]] = None
        self._initialized = False

    def initialize(self, training_dir: Optional[str] = None):
        """预计算训练集分布（服务启动时调用）"""
        if self._initialized:
            return

        training_dir = training_dir or self.config.get("data", {}).get("training_dir", "")
        if not training_dir or not os.path.exists(training_dir):
            logger.warning(f"训练集目录不存在: {training_dir}")
            return

        logger.info("=" * 60)
        logger.info("预计算训练集分布")
        logger.info("=" * 60)

        t_start = time.time()

        # 加载训练集
        train_annotations = load_annotations(training_dir)
        train_image_paths = [a.image_path for a in train_annotations]
        logger.info(f"  训练集: {len(train_annotations)} 张图片")

        # 类别分析
        from analyzers.class_analyzer import ClassAnalyzer
        class_analyzer = ClassAnalyzer(self.config)
        train_class_dist = compute_class_distribution(train_annotations)
        # 类别分析只需要训练集
        self.train_class_analysis = class_analyzer.analyze(train_class_dist, [])
        logger.info(f"  类别分析完成: {len(self.train_class_analysis.get('class_scores', {}))} 个类别")

        # 语义维度分析
        from analyzers import (
            LightingAnalyzer,
            ViewpointAnalyzer,
            BlurAnalyzer,
            WeatherAnalyzer,
            TimeOfDayAnalyzer,
            EnvironmentAnalyzer,
        )

        semantic_config = self.config.get("semantic", {})
        analyzers = {
            "lighting": LightingAnalyzer(semantic_config),
            "viewpoint": ViewpointAnalyzer(semantic_config),
            "blur": BlurAnalyzer(semantic_config),
            "weather": WeatherAnalyzer(semantic_config),
            "timeOfDay": TimeOfDayAnalyzer(semantic_config),
            "environment": EnvironmentAnalyzer(semantic_config),
        }

        self.train_semantic_distribution = {}
        for dim_name, analyzer in analyzers.items():
            logger.info(f"  分析训练集 [{dim_name}]...")
            results = analyzer.analyze_batch(train_image_paths)
            distribution = analyzer.compute_distribution(results)
            self.train_semantic_distribution[dim_name] = distribution
            logger.info(f"    分布: {distribution.category_percentages}")

        self._initialized = True
        duration = time.time() - t_start
        logger.info(f"训练集分布预计算完成，耗时: {duration:.2f}s")

    def analyze_single(
        self,
        image_path: str,
        detections: List[Dict[str, Any]],
        false_positives: Optional[List[Dict[str, Any]]] = None,
        false_negatives: Optional[List[Dict[str, Any]]] = None,
        verification_result: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """单图归因分析

        Args:
            image_path: 图片路径
            detections: 检测结果列表
            false_positives: 误报列表 [{"class_name": "...", "confidence": 0.8, "reason": "..."}]
            false_negatives: 漏报列表 [{"class_name": "...", "reason": "..."}]
            verification_result: detection-service 的校验结果

        Returns:
            归因分析结果
        """
        false_positives = false_positives or []
        false_negatives = false_negatives or []
        if not self._initialized:
            self.initialize()

        from analyzers import (
            LightingAnalyzer,
            ViewpointAnalyzer,
            BlurAnalyzer,
            WeatherAnalyzer,
            TimeOfDayAnalyzer,
            EnvironmentAnalyzer,
            LLMAttributionAnalyzer,
        )

        # 6 个分析器分析该图片
        semantic_config = self.config.get("semantic", {})
        analyzers = {
            "lighting": LightingAnalyzer(semantic_config),
            "viewpoint": ViewpointAnalyzer(semantic_config),
            "blur": BlurAnalyzer(semantic_config),
            "weather": WeatherAnalyzer(semantic_config),
            "timeOfDay": TimeOfDayAnalyzer(semantic_config),
            "environment": EnvironmentAnalyzer(semantic_config),
        }

        semantic_dimensions: Dict[str, SemanticDimensionResult] = {}
        for dim_name, analyzer in analyzers.items():
            result = analyzer.analyze_single(image_path)
            semantic_dimensions[dim_name] = result

        # LLM 归因分析
        from config import get_llm_config
        llm_config = get_llm_config()
        llm_analyzer = LLMAttributionAnalyzer(llm_config)

        attribution = llm_analyzer.attribute(
            image_path=image_path,
            detections=detections,
            class_analysis=self.train_class_analysis or {},
            semantic_dimensions=semantic_dimensions,
            train_distribution=self.train_semantic_distribution or {},
            false_positives=false_positives,
            false_negatives=false_negatives,
            verification_result=verification_result,
        )

        return {
            "filename": attribution.filename,
            "attribution_type": attribution.attribution_type.value,
            "confidence": attribution.confidence,
            "reasoning": attribution.reasoning,
            "main_cause_dimension": attribution.main_cause_dimension,
            "dimension_attributions": [
                {
                    "dimension": da.dimension,
                    "category": da.category,
                    "train_coverage": da.train_coverage,
                    "is_gap": da.is_gap,
                    "contribution": da.contribution,
                }
                for da in attribution.dimension_attributions
            ],
            "feedback_suggestion": attribution.feedback_suggestion,
            "should_feedback": attribution.should_feedback,
        }

    def run_full(
        self,
        training_dir: Optional[str] = None,
        test_dir: Optional[str] = None,
    ) -> dict:
        """批量分析（保留兼容性）"""
        # 预计算训练集分布
        self.initialize(training_dir)

        test_dir = test_dir or self.config.get("data", {}).get("test_dir", "")
        if not test_dir or not os.path.exists(test_dir):
            return {"error": "测试集目录不存在"}

        from data_loader import load_annotations
        test_annotations = load_annotations(test_dir)

        results = []
        for ann in test_annotations:
            detections = [
                {
                    "class_id": b.class_id,
                    "class_name": b.class_name,
                    "confidence": b.confidence,
                }
                for b in ann.bboxes
            ]

            # 判断是否是误报（没有检测结果 = 漏报）
            is_false_positive = len(detections) > 0

            result = self.analyze_single(
                image_path=ann.image_path,
                detections=detections,
                is_false_positive=is_false_positive,
            )
            results.append(result)

        return {
            "training_dir": self.config.get("data", {}).get("training_dir", ""),
            "test_dir": test_dir,
            "test_image_count": len(test_annotations),
            "attribution_results": results,
            "summary": {
                "total": len(results),
                "should_feedback": sum(1 for r in results if r.get("should_feedback")),
                "should_not_feedback": sum(1 for r in results if not r.get("should_feedback")),
            }
        }

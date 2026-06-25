"""LLM 归因分析器 — 基于多维度分析结果进行误报/漏报归因"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from schemas import (
    AttributionType,
    DatasetSemanticDistribution,
    DimensionAttribution,
    LLMAttribution,
    SemanticDimensionResult,
)

logger = logging.getLogger(__name__)


class LLMAttributionAnalyzer:
    """LLM 归因分析器

    接收所有维度的分析结果，使用大模型分析误报/漏报的原因。
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.protocol = config.get("protocol", "openai")
        self.timeout = config.get("timeout", 300)
        self.temperature = config.get("temperature", 0.1)
        self._client = None

    def _get_client(self):
        """获取 LLM 客户端"""
        if self._client is not None:
            return self._client

        try:
            import openai

            if self.protocol == "openai":
                provider_config = self.config.get("openai", {})
                self._client = openai.OpenAI(
                    api_key=provider_config.get("api_key", ""),
                    base_url=provider_config.get("api_url", "https://api.openai.com/v1"),
                    timeout=self.timeout,
                )
            elif self.protocol == "ollama":
                provider_config = self.config.get("ollama", {})
                self._client = openai.OpenAI(
                    api_key="ollama",
                    base_url=provider_config.get("api_url", "http://localhost:11434/v1"),
                    timeout=self.timeout,
                )
            else:
                raise ValueError(f"不支持的协议: {self.protocol}")

            return self._client
        except Exception as e:
            logger.error(f"创建 LLM 客户端失败: {e}")
            return None

    def _build_prompt(
        self,
        image_path: str,
        detections: List[Dict[str, Any]],
        class_analysis: Dict[str, Any],
        semantic_dimensions: Dict[str, SemanticDimensionResult],
        train_distribution: Dict[str, DatasetSemanticDistribution],
        false_positives: Optional[List[Dict[str, Any]]] = None,
        false_negatives: Optional[List[Dict[str, Any]]] = None,
        verification_result: Optional[Dict[str, Any]] = None,
    ) -> str:
        """构建分析提示词"""
        false_positives = false_positives or []
        false_negatives = false_negatives or []

        # 问题类型描述
        problem_types = []
        if false_positives:
            problem_types.append("误报")
        if false_negatives:
            problem_types.append("漏报")

        if problem_types:
            problem_type_desc = f"问题类型: {' + '.join(problem_types)}"
        else:
            problem_type_desc = "问题类型: 未知（需要分析）"

        # 检测结果描述
        detection_desc = "当前检测结果:\n"
        if detections:
            for det in detections[:5]:
                detection_desc += f"- 类别: {det.get('class_name', 'unknown')}, 置信度: {det.get('confidence', 0):.2f}\n"
        else:
            detection_desc += "  未检测到目标\n"

        # 误报描述
        fp_desc = ""
        if false_positives:
            fp_desc = "\n误报列表（模型错误检测到的目标）:\n"
            for fp in false_positives:
                fp_desc += f"- 类别: {fp.get('class_name', 'unknown')}"
                if fp.get('confidence'):
                    fp_desc += f", 置信度: {fp['confidence']:.2f}"
                if fp.get('reason'):
                    fp_desc += f", 原因: {fp['reason']}"
                fp_desc += "\n"

        # 漏报描述
        fn_desc = ""
        if false_negatives:
            fn_desc = "\n漏报列表（模型未检测到的目标）:\n"
            for fn in false_negatives:
                fn_desc += f"- 类别: {fn.get('class_name', 'unknown')}"
                if fn.get('reason'):
                    fn_desc += f", 原因: {fn['reason']}"
                fn_desc += "\n"

        # 校验服务结果描述
        verification_desc = ""
        if verification_result:
            verification_desc = f"""
校验服务分析:
- 校验结果: {verification_result.get('result', 'unknown')}
- 校验置信度: {verification_result.get('confidence', 0):.2f}
- 校验说明: {verification_result.get('reason', '无')}
"""

        # 类别覆盖分析
        class_desc = "训练集类别覆盖分析:\n"
        coverage_gaps = class_analysis.get("coverage_gaps", [])
        overrepresented = class_analysis.get("overrepresented_in_test", [])

        if coverage_gaps:
            class_desc += "覆盖薄弱类别（训练集占比 < 2%）:\n"
            for gap in coverage_gaps:
                class_desc += f"  - {gap.get('class_name', '')}: 训练集占比 {gap.get('train_pct', 0)}%\n"

        if overrepresented:
            class_desc += "测试集占比异常高的类别:\n"
            for over in overrepresented:
                class_desc += f"  - {over.get('class_name', '')}: 测试集占比 {over.get('test_pct', 0)}%, 训练集占比 {over.get('train_pct', 0)}%\n"

        if not coverage_gaps and not overrepresented:
            class_desc += "  类别覆盖情况良好\n"

        # 语义维度分析
        semantic_desc = "训练集语义维度分布分析:\n"
        dimension_names = {
            "lighting": "光照",
            "viewpoint": "视角",
            "blur": "清晰度",
            "weather": "天气",
            "timeOfDay": "时间",
            "environment": "环境",
        }

        for dim_key, dim_name in dimension_names.items():
            dim_result = semantic_dimensions.get(dim_key)
            train_dist = train_distribution.get(dim_key)

            if dim_result and train_dist:
                best_cat = dim_result.best_category
                train_pct = train_dist.category_percentages.get(best_cat, 0)

                # 判断是否是覆盖缺口
                is_gap = train_pct < 0.05
                gap_marker = " ⚠️ 覆盖不足" if is_gap else ""

                semantic_desc += f"- {dim_name}: {best_cat} (置信度: {dim_result.confidence:.2f})"
                semantic_desc += f" | 训练集占比: {train_pct:.1%}{gap_marker}\n"

                # 显示该维度的完整分布
                semantic_desc += f"  训练集分布: {train_dist.category_percentages}\n"

        prompt = f"""你是一个专业的计算机视觉数据分析专家。请分析以下误报/漏报的根本原因。

## 图片信息
图片路径: {image_path}

## {problem_type_desc}

{detection_desc}
{fp_desc}
{fn_desc}
{verification_desc}

## {class_desc}

## {semantic_desc}

## 分析任务
请根据以上信息，分析该图片被标记为误报/漏报的根本原因。

重点分析:
1. 哪个维度（类别、光照、视角、清晰度、天气、时间、环境）是最主要的原因？
2. 该维度在训练集中的覆盖情况如何？
3. 该维度的覆盖不足如何导致了误报/漏报？
4. 建议如何补充训练数据来解决这个问题？

请用 JSON 格式返回分析结果:
{{
    "attribution_type": "归因类型(环境因素/类间混淆/遮挡截断/背景干扰/标注错误/其他)",
    "confidence": 0.0-1.0,
    "reasoning": "详细分析原因，说明哪个维度是主因，以及该维度如何导致误报/漏报",
    "dimension_attributions": [
        {{
            "dimension": "维度名称(lighting/viewpoint/blur/weather/timeOfDay/environment/class)",
            "category": "该图片在该维度的分类",
            "train_coverage": 0.0-1.0,
            "is_gap": true/false,
            "contribution": "该维度对误报/漏报的贡献描述"
        }}
    ],
    "main_cause_dimension": "主要原因维度",
    "feedback_suggestion": "回流建议，说明应该补充哪类数据",
    "should_feedback": true/false
}}"""

        return prompt

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """解析 LLM 响应"""
        try:
            cleaned = re.sub(r'```json\s*', '', response_text)
            cleaned = re.sub(r'```\s*$', '', cleaned)

            result = json.loads(cleaned)

            attribution_type = result.get("attribution_type", "其他")
            try:
                attr_enum = AttributionType(attribution_type)
            except ValueError:
                attr_enum = AttributionType.OTHER

            dimension_attributions = []
            for da in result.get("dimension_attributions", []):
                dimension_attributions.append(DimensionAttribution(
                    dimension=da.get("dimension", ""),
                    category=da.get("category", ""),
                    train_coverage=float(da.get("train_coverage", 0)),
                    is_gap=bool(da.get("is_gap", False)),
                    contribution=str(da.get("contribution", "")),
                ))

            return {
                "attribution_type": attr_enum,
                "confidence": min(1.0, max(0.0, float(result.get("confidence", 0.5)))),
                "reasoning": str(result.get("reasoning", "")),
                "dimension_attributions": dimension_attributions,
                "main_cause_dimension": str(result.get("main_cause_dimension", "")),
                "feedback_suggestion": str(result.get("feedback_suggestion", "")),
                "should_feedback": bool(result.get("should_feedback", True)),
            }
        except Exception as e:
            logger.warning(f"解析 LLM 响应失败: {e}")
            return {
                "attribution_type": AttributionType.OTHER,
                "confidence": 0.3,
                "reasoning": f"解析失败: {response_text[:200]}",
                "dimension_attributions": [],
                "main_cause_dimension": "unknown",
                "feedback_suggestion": "需要人工复核",
                "should_feedback": True,
            }

    def attribute(
        self,
        image_path: str,
        detections: List[Dict[str, Any]],
        class_analysis: Dict[str, Any],
        semantic_dimensions: Dict[str, SemanticDimensionResult],
        train_distribution: Dict[str, DatasetSemanticDistribution],
        false_positives: Optional[List[Dict[str, Any]]] = None,
        false_negatives: Optional[List[Dict[str, Any]]] = None,
        verification_result: Optional[Dict[str, Any]] = None,
    ) -> LLMAttribution:
        """进行归因分析"""
        filename = Path(image_path).name

        prompt = self._build_prompt(
            image_path, detections, class_analysis,
            semantic_dimensions, train_distribution,
            false_positives, false_negatives, verification_result
        )

        client = self._get_client()
        if client is None:
            return LLMAttribution(
                filename=filename,
                attribution_type=AttributionType.OTHER,
                confidence=0.0,
                reasoning="LLM 客户端不可用",
                dimension_attributions=[],
                main_cause_dimension="unknown",
                feedback_suggestion="需要人工复核",
                should_feedback=True,
            )

        try:
            model_name = self.config.get("openai", {}).get("model", "gpt-4o")

            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=2000,
                temperature=self.temperature,
            )

            response_text = response.choices[0].message.content
            result = self._parse_response(response_text)

            return LLMAttribution(
                filename=filename,
                **result
            )

        except Exception as e:
            logger.error(f"LLM 归因分析失败: {e}")
            return LLMAttribution(
                filename=filename,
                attribution_type=AttributionType.OTHER,
                confidence=0.0,
                reasoning=f"分析失败: {str(e)}",
                dimension_attributions=[],
                main_cause_dimension="unknown",
                feedback_suggestion="需要人工复核",
                should_feedback=True,
            )

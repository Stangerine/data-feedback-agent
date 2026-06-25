#!/usr/bin/env python3
"""测试归因大模型效果"""

import json
import sys
import os

# 添加路径 (从 tests/ 目录向上找到 services)
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(parent_dir, "services", "data-analysis-service"))

from config import get_config, get_llm_config
from pipeline import AnalysisPipeline


def test_single_image():
    """测试单张图片的归因分析"""
    print("=" * 60)
    print("测试归因大模型效果")
    print("=" * 60)

    # 加载配置
    config = get_config()
    llm_config = get_llm_config()

    print(f"\nLLM 配置:")
    print(f"  API URL: {llm_config['openai']['api_url']}")
    print(f"  Model: {llm_config['openai']['model']}")

    # 创建 pipeline
    pipeline = AnalysisPipeline(config)

    # 初始化（预计算训练集分布）
    print("\n初始化训练集分布...")
    pipeline.initialize()
    print("初始化完成!")

    # 测试图片
    test_images = [
        "E:\\zzq\\误报\\20250307155309293.jpg",
        "E:\\zzq\\误报\\20250311140729961.jpg",
        "E:\\zzq\\误报\\20250313113730400.jpg",
    ]

    for image_path in test_images:
        if not os.path.exists(image_path):
            print(f"\n图片不存在: {image_path}")
            continue

        print(f"\n{'=' * 60}")
        print(f"分析图片: {os.path.basename(image_path)}")
        print("=" * 60)

        # 模拟误报情况
        false_positives = [
            {
                "class_name": "wajueji",
                "confidence": 0.75,
                "reason": "背景干扰导致误检测"
            }
        ]

        # 进行归因分析
        result = pipeline.analyze_single(
            image_path=image_path,
            detections=[],
            false_positives=false_positives,
            false_negatives=[],
        )

        # 输出结果
        print(f"\n归因结果:")
        print(f"  归因类型: {result.get('attribution_type')}")
        print(f"  置信度: {result.get('confidence'):.2f}")
        print(f"  主因维度: {result.get('main_cause_dimension')}")
        print(f"  是否建议回流: {result.get('should_feedback')}")
        print(f"  回流建议: {result.get('feedback_suggestion')}")

        print(f"\n  详细分析:")
        reasoning = result.get('reasoning', '')
        if reasoning:
            # 每行缩进显示
            for line in reasoning.split('\n'):
                if line.strip():
                    print(f"    {line}")

        print(f"\n  各维度归因:")
        for da in result.get('dimension_attributions', []):
            print(f"    - {da.get('dimension')}: {da.get('category')}")
            print(f"      训练集占比: {da.get('train_coverage'):.1%}")
            print(f"      覆盖缺口: {da.get('is_gap')}")
            print(f"      贡献: {da.get('contribution')}")


if __name__ == "__main__":
    test_single_image()

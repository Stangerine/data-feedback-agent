#!/usr/bin/env python3
"""
数据分析服务测试

测试流程：
1. 加载训练集分布
2. 对误报图片进行归因分析
"""

import os
import sys
import time
import json

# 添加路径
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(parent_dir, "services", "data-analysis-service"))

def test_data_analysis():
    """测试数据分析服务"""
    print("=" * 70)
    print("数据分析服务测试")
    print("=" * 70)

    # 1. 加载配置
    print("\n[1] 加载配置...")
    from config import get_config
    cfg = get_config()

    training_dir = cfg.get("data", {}).get("training_dir", "")
    test_dir = cfg.get("data", {}).get("test_dir", "")

    print(f"  训练集: {training_dir}")
    print(f"  测试集: {test_dir}")

    # 检查目录
    if not os.path.exists(training_dir):
        print(f"  [ERROR] 训练集目录不存在: {training_dir}")
        return
    if not os.path.exists(test_dir):
        print(f"  [ERROR] 测试集目录不存在: {test_dir}")
        return

    # 2. 获取测试图片
    print("\n[2] 获取测试图片...")
    image_files = sorted([
        f for f in os.listdir(test_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
    ])[:3]  # 只测试前3张

    print(f"  测试图片: {len(image_files)} 张")
    for img in image_files:
        print(f"    - {img}")

    # 3. 初始化 Pipeline
    print("\n[3] 初始化分析 Pipeline...")
    from pipeline import AnalysisPipeline

    pipeline = AnalysisPipeline(cfg)

    print("  预计算训练集分布 (首次需要较长时间)...")
    t0 = time.time()
    pipeline.initialize()
    duration = time.time() - t0
    print(f"  初始化完成，耗时: {duration:.2f}s")

    # 4. 测试单图归因分析
    print("\n[4] 测试单图归因分析...")

    for img_name in image_files:
        img_path = os.path.join(test_dir, img_name)
        print(f"\n{'=' * 70}")
        print(f"分析图片: {img_name}")
        print('=' * 70)

        # 模拟检测结果 (假设全部是误报)
        detections = [
            {"class_name": "wajueji", "confidence": 0.75, "class_id": 0},
        ]

        false_positives = [
            {"class_name": "wajueji", "confidence": 0.75, "reason": "背景干扰导致误检测"}
        ]

        try:
            t1 = time.time()
            result = pipeline.analyze_single(
                image_path=img_path,
                detections=detections,
                false_positives=false_positives,
                false_negatives=[],
            )
            duration = time.time() - t1

            print(f"\n[结果]")
            print(f"  归因类型: {result.get('attribution_type')}")
            print(f"  置信度: {result.get('confidence', 0):.2f}")
            print(f"  主因维度: {result.get('main_cause_dimension')}")
            print(f"  是否建议回流: {result.get('should_feedback')}")
            print(f"  回流建议: {result.get('feedback_suggestion', '')[:150]}...")
            print(f"  耗时: {duration:.2f}s")

            print(f"\n  各维度归因:")
            for da in result.get('dimension_attributions', []):
                print(f"    - {da.get('dimension')}: {da.get('category')}")
                print(f"      训练集占比: {da.get('train_coverage', 0):.1%}")
                print(f"      覆盖缺口: {da.get('is_gap')}")

        except Exception as e:
            print(f"  [ERROR] 分析失败: {e}")
            import traceback
            traceback.print_exc()

    # 5. 统计训练集分布
    print("\n" + "=" * 70)
    print("[5] 训练集分布统计")
    print("=" * 70)

    if pipeline.train_class_analysis:
        class_scores = pipeline.train_class_analysis.get('class_scores', {})
        print(f"\n  类别分布:")
        for cls_name, score in class_scores.items():
            print(f"    - {cls_name}: {score:.2%}")

    if pipeline.train_semantic_distribution:
        print(f"\n  语义维度分布:")
        for dim_name, dist in pipeline.train_semantic_distribution.items():
            print(f"    [{dim_name}]")
            for cat, pct in dist.category_percentages.items():
                print(f"      - {cat}: {pct:.1%}")

    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)


if __name__ == "__main__":
    test_data_analysis()

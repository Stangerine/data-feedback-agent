#!/usr/bin/env python3
"""测试归因大模型 — 使用真实误报数据，带调试信息"""

import json
import os
import sys
import xml.etree.ElementTree as ET

# 添加父目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import get_config, get_llm_config
from data_loader import CLASS_NAME_TO_ID
from pipeline import AnalysisPipeline


def load_false_positives_from_xml(xml_dir: str) -> list:
    """从 XML 标注加载误报数据"""
    results = []
    if not os.path.exists(xml_dir):
        return results

    for fname in sorted(os.listdir(xml_dir)):
        if not fname.endswith(".xml"):
            continue

        xml_path = os.path.join(xml_dir, fname)
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            filename = root.find("filename").text
            image_path = os.path.join(os.path.dirname(xml_dir), "images", filename)

            if not os.path.exists(image_path):
                continue

            # 提取真实标注
            objects = []
            for obj in root.findall("object"):
                name = obj.find("name").text
                bndbox = obj.find("bndbox")
                xmin = int(bndbox.find("xmin").text)
                ymin = int(bndbox.find("ymin").text)
                xmax = int(bndbox.find("xmax").text)
                ymax = int(bndbox.find("ymax").text)
                objects.append({
                    "class_name": name,
                    "bbox": [xmin, ymin, xmax, ymax],
                })

            results.append({
                "image_path": image_path,
                "xml_path": xml_path,
                "filename": filename,
                "ground_truth": objects,
            })
        except Exception as e:
            print(f"解析 {fname} 失败: {e}")

    return results


def test_single_attribution():
    """测试单张图片的归因分析"""
    print("=" * 70)
    print("归因大模型测试 — 真实误报数据（带调试信息）")
    print("=" * 70)

    # 加载配置
    config = get_config()
    llm_config = get_llm_config()

    print(f"\n[配置信息]")
    print(f"  LLM 模型: {llm_config['openai']['model']}")
    print(f"  LLM API: {llm_config['openai']['api_url']}")
    print(f"  训练集: {config.get('data', {}).get('training_dir', 'N/A')}")
    print(f"  测试集: {config.get('data', {}).get('test_dir', 'N/A')}")

    semantic_config = config.get("semantic", {})
    print(f"\n[语义模型配置]")
    print(f"  模型路径: {semantic_config.get('model_name', 'N/A')}")
    print(f"  设备: {semantic_config.get('device', 'N/A')}")
    print(f"  Batch Size: {semantic_config.get('batch_size', 'N/A')}")

    # 创建 pipeline
    pipeline = AnalysisPipeline(config)

    # 初始化（预计算训练集分布）
    print(f"\n[初始化] 正在预计算训练集分布...")
    import time
    t_start = time.time()
    pipeline.initialize()
    t_init = time.time() - t_start
    print(f"[初始化] 完成! 耗时: {t_init:.2f}s")

    # 检查训练集分布
    if pipeline.train_class_analysis:
        print(f"\n[训练集类别分析]")
        print(f"  覆盖缺口: {pipeline.train_class_analysis.get('coverage_gaps', [])}")
        print(f"  测试集过表示: {pipeline.train_class_analysis.get('overrepresented_in_test', [])}")

    if pipeline.train_semantic_distribution:
        print(f"\n[训练集语义分布]")
        for dim_name, dist in pipeline.train_semantic_distribution.items():
            print(f"  {dim_name}: {dist.category_percentages}")

    # 加载误报数据
    test_dir = config.get("data", {}).get("test_dir", "")
    xml_dir = os.path.join(test_dir, "xml")
    print(f"\n[加载数据] 误报目录: {xml_dir}")

    fp_data = load_false_positives_from_xml(xml_dir)
    print(f"[加载数据] 找到 {len(fp_data)} 张误报图片")

    if not fp_data:
        print("[错误] 没有找到误报数据!")
        return

    # 测试第 1 张图片
    print(f"\n{'=' * 70}")
    print(f"[测试] 分析第 1 张图片")
    print("=" * 70)

    item = fp_data[0]
    print(f"  文件名: {item['filename']}")
    print(f"  图片路径: {item['image_path']}")
    print(f"  真实标注: {json.dumps(item['ground_truth'], ensure_ascii=False, indent=4)}")

    # 构建漏报列表
    false_negatives = [
        {"class_name": obj["class_name"], "reason": "模型未检测到"}
        for obj in item["ground_truth"]
    ]
    print(f"  漏报列表: {json.dumps(false_negatives, ensure_ascii=False)}")

    # 逐步分析
    print(f"\n[步骤 1] 分析语义维度...")
    from analyzers import (
        LightingAnalyzer,
        ViewpointAnalyzer,
        BlurAnalyzer,
        WeatherAnalyzer,
        TimeOfDayAnalyzer,
        EnvironmentAnalyzer,
    )

    analyzers = {
        "lighting": LightingAnalyzer(semantic_config),
        "viewpoint": ViewpointAnalyzer(semantic_config),
        "blur": BlurAnalyzer(semantic_config),
        "weather": WeatherAnalyzer(semantic_config),
        "timeOfDay": TimeOfDayAnalyzer(semantic_config),
        "environment": EnvironmentAnalyzer(semantic_config),
    }

    from schemas import SemanticDimensionResult
    semantic_dimensions = {}

    for dim_name, analyzer in analyzers.items():
        print(f"  分析 {dim_name}...")
        t_start = time.time()
        result = analyzer.analyze_single(item["image_path"])
        t_dim = time.time() - t_start
        semantic_dimensions[dim_name] = result
        print(f"    结果: {result.best_category} (置信度: {result.confidence:.4f}, 耗时: {t_dim:.2f}s)")
        if result.similarities:
            print(f"    相似度分布: { {k: round(v, 4) for k, v in result.similarities.items()} }")

    print(f"\n[步骤 2] LLM 归因分析...")
    from analyzers import LLMAttributionAnalyzer

    llm_analyzer = LLMAttributionAnalyzer(llm_config)

    # 打印 prompt（调试用）
    print(f"  构建 prompt...")
    prompt = llm_analyzer._build_prompt(
        image_path=item["image_path"],
        detections=[],
        class_analysis=pipeline.train_class_analysis or {},
        semantic_dimensions=semantic_dimensions,
        train_distribution=pipeline.train_semantic_distribution or {},
        false_positives=[],
        false_negatives=false_negatives,
    )
    print(f"  Prompt 长度: {len(prompt)} 字符")
    print(f"  Prompt 预览:")
    print("-" * 60)
    print(prompt[:1500])
    print("... (省略)")
    print("-" * 60)

    print(f"\n[步骤 3] 调用 LLM API...")
    t_start = time.time()
    result = pipeline.analyze_single(
        image_path=item["image_path"],
        detections=[],
        false_positives=[],
        false_negatives=false_negatives,
    )
    t_llm = time.time() - t_start
    print(f"  LLM 调用耗时: {t_llm:.2f}s")

    # 输出结果
    print(f"\n[归因结果]")
    print(f"  归因类型: {result.get('attribution_type')}")
    print(f"  置信度: {result.get('confidence'):.2f}")
    print(f"  主因维度: {result.get('main_cause_dimension')}")
    print(f"  建议回流: {result.get('should_feedback')}")
    print(f"  回流建议: {result.get('feedback_suggestion')}")

    reasoning = result.get('reasoning', '')
    if reasoning:
        print(f"\n[详细分析]")
        for line in reasoning.split('\n'):
            if line.strip():
                print(f"  {line}")

    print(f"\n[各维度归因]")
    for da in result.get('dimension_attributions', []):
        print(f"  - {da.get('dimension')}: {da.get('category')}")
        print(f"    训练集占比: {da.get('train_coverage'):.1%}")
        print(f"    覆盖缺口: {da.get('is_gap')}")
        print(f"    贡献: {da.get('contribution')}")


if __name__ == "__main__":
    test_single_attribution()

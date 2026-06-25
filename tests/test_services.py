#!/usr/bin/env python3
"""
服务逻辑测试

按照 detection-service 和 data-analysis-service 的 API 逻辑进行测试
"""

import os
import sys
import time
import json
import subprocess

# 路径
parent_dir = os.path.dirname(os.path.dirname(__file__))
det_service_dir = os.path.join(parent_dir, "services", "detection-service")
analysis_service_dir = os.path.join(parent_dir, "services", "data-analysis-service")


def test_detection_service():
    """测试 detection-service 逻辑 (独立进程)"""
    print("=" * 70)
    print("[detection-service] 检测校验服务测试")
    print("=" * 70)

    test_script = '''
import os
import sys
import time
import json

sys.path.insert(0, r"{det_dir}")

from config import get_config
from detection import DetectionClient
from verifier import Verifier
from services.correction_service import CorrectionService

cfg = get_config()
test_dir = cfg.data.test_dir

print(f"配置信息:")
print(f"  检测 API: {{cfg.detection.api_url}}")
print(f"  LLM 模型: {{cfg.llm.model}}")
print(f"  测试目录: {{test_dir}}")

# 获取测试图片
image_files = sorted([
    f for f in os.listdir(test_dir)
    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
])[:3]

print(f"  测试图片: {{len(image_files)}} 张")

# 初始化组件
det_client = DetectionClient(
    api_url=cfg.detection.api_url,
    model_id=cfg.detection.model_id,
    box_threshold=cfg.detection.box_threshold,
    timeout=cfg.detection.timeout,
)
verifier = Verifier()
correction_service = CorrectionService()

results = []

for img_name in image_files:
    img_path = os.path.join(test_dir, img_name)
    print(f"\\n{{'─' * 70}}")
    print(f"处理图片: {{img_name}}")
    print('─' * 70)

    # 步骤 1: 小模型检测
    print(f"\\n  [步骤1] 小模型检测...", end=" ", flush=True)
    try:
        t0 = time.time()
        detections = det_client.detect(img_path, cfg.detection.box_threshold)
        detect_time = time.time() - t0
        print(f"OK ({{detect_time:.2f}}s)")
        print(f"    检测到 {{len(detections)}} 个目标:")
        for i, det in enumerate(detections, 1):
            print(f"      {{i}}. {{det['class_name']}} ({{det['confidence']:.2f}})")
    except Exception as e:
        print(f"FAIL: {{e}}")
        results.append({{"image": img_name, "error": f"检测失败: {{e}}"}})
        continue

    # 步骤 2: 大模型校验
    print(f"  [步骤2] 大模型校验...", end=" ", flush=True)
    try:
        t1 = time.time()
        verification = verifier.verify(img_path, detections)
        verify_time = time.time() - t1
        print(f"OK ({{verify_time:.2f}}s)")

        if verification["success"]:
            data = verification["data"]
            fps = data.get("false_positives", [])
            mds = data.get("missed_detections", [])
            overall = data.get("overall_assessment", {{}})
            print(f"    误报: {{len(fps)}} 个")
            for fp in fps:
                print(f"      - {{fp.get('reported_class')}} -> {{fp.get('actual_class')}}")
            print(f"    漏报: {{len(mds)}} 个")
            for md in mds:
                print(f"      - {{md.get('actual_class_cn', md.get('actual_class'))}}")
            print(f"    质量: {{overall.get('detection_quality', '?')}}")
        else:
            print(f"    校验失败: {{verification.get('error')}}")
    except Exception as e:
        print(f"FAIL: {{e}}")
        verification = {{"success": False, "error": str(e)}}

    # 步骤 3: 自动纠正
    print(f"  [步骤3] 自动纠正...", end=" ", flush=True)
    try:
        t2 = time.time()
        correction = correction_service.correct(img_path, detections, verification)
        correct_time = time.time() - t2
        print(f"OK ({{correct_time:.2f}}s)")

        artifacts = correction.get("artifacts", {{}})
        print(f"    输出目录: {{artifacts.get('dir', '')}}")
    except Exception as e:
        print(f"FAIL: {{e}}")
        correction = {{"error": str(e)}}

    results.append({{
        "image": img_name,
        "detections": detections,
        "verification": verification,
        "correction": correction,
    }})

# 保存结果
output_file = r"{parent_dir}\\tests\\detection_results.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=str)
print(f"\\n结果已保存: {{output_file}}")
'''.format(det_dir=det_service_dir, parent_dir=parent_dir)

    # 写入临时脚本
    temp_script = os.path.join(parent_dir, "tests", "_test_det.py")
    with open(temp_script, "w", encoding="utf-8") as f:
        f.write(test_script)

    # 运行
    result = subprocess.run([sys.executable, temp_script], capture_output=False)
    return result.returncode == 0


def test_data_analysis_service():
    """测试 data-analysis-service 逻辑 (独立进程)"""
    print("\n" + "=" * 70)
    print("[data-analysis-service] 数据分析服务测试")
    print("=" * 70)

    test_script = '''
import os
import sys
import time
import json

sys.path.insert(0, r"{analysis_dir}")

from config import get_config
from pipeline import AnalysisPipeline

cfg = get_config()
training_dir = cfg.get("data", {{}}).get("training_dir", "")
test_dir = cfg.get("data", {{}}).get("test_dir", "")

print(f"配置信息:")
print(f"  训练集: {{training_dir}}")
print(f"  测试集: {{test_dir}}")

# 初始化 Pipeline
print(f"\\n初始化分析 Pipeline...")
pipeline = AnalysisPipeline(cfg)

print(f"预计算训练集分布...")
t0 = time.time()
pipeline.initialize()
init_time = time.time() - t0
print(f"初始化完成 ({{init_time:.2f}}s)")

# 获取测试图片
image_files = sorted([
    f for f in os.listdir(test_dir)
    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
])[:3]

print(f"\\n测试图片: {{len(image_files)}} 张")

# 加载检测结果 (如果有)
det_results_file = r"{parent_dir}\\tests\\detection_results.json"
detection_results = {{}}
if os.path.exists(det_results_file):
    with open(det_results_file, "r", encoding="utf-8") as f:
        det_results = json.load(f)
        for item in det_results:
            detection_results[item["image"]] = item

# 归因分析
attribution_results = []

for img_name in image_files:
    img_path = os.path.join(test_dir, img_name)
    print(f"\\n{{'─' * 70}}")
    print(f"归因分析: {{img_name}}")
    print('─' * 70)

    # 从检测结果提取信息
    det_item = detection_results.get(img_name, {{}})
    detections = det_item.get("detections", [])
    verification = det_item.get("verification", {{}})

    false_positives = []
    false_negatives = []

    if verification.get("success"):
        data = verification["data"]
        false_positives = data.get("false_positives", [])
        false_negatives = data.get("missed_detections", [])

    print(f"  检测数: {{len(detections)}}")
    print(f"  误报数: {{len(false_positives)}}")
    print(f"  漏报数: {{len(false_negatives)}}")

    try:
        t1 = time.time()
        result = pipeline.analyze_single(
            image_path=img_path,
            detections=detections,
            false_positives=false_positives,
            false_negatives=false_negatives,
            verification_result=verification,
        )
        analyze_time = time.time() - t1

        print(f"\\n  [归因结果] ({{analyze_time:.2f}}s)")
        print(f"    归因类型: {{result.get('attribution_type')}}")
        print(f"    置信度: {{result.get('confidence', 0):.2f}}")
        print(f"    主因维度: {{result.get('main_cause_dimension')}}")
        print(f"    是否建议回流: {{result.get('should_feedback')}}")
        print(f"    回流建议: {{result.get('feedback_suggestion', '')[:100]}}...")

        print(f"\\n    各维度归因:")
        for da in result.get('dimension_attributions', []):
            print(f"      - {{da.get('dimension')}}: {{da.get('category')}}")
            print(f"        训练集占比: {{da.get('train_coverage', 0):.1%}}")
            print(f"        覆盖缺口: {{da.get('is_gap')}}")

        attribution_results.append({{
            "image": img_name,
            "attribution": result,
        }})
    except Exception as e:
        print(f"  [ERROR] 归因失败: {{e}}")
        import traceback
        traceback.print_exc()

# 统计训练集分布
print(f"\\n{{'=' * 70}}")
print("训练集分布统计")
print('=' * 70)

if pipeline.train_class_analysis:
    class_scores = pipeline.train_class_analysis.get('class_scores', {{}})
    print(f"\\n  类别分布:")
    for cls_name, score in class_scores.items():
        print(f"    - {{cls_name}}: {{score:.2%}}")

if pipeline.train_semantic_distribution:
    print(f"\\n  语义维度分布:")
    for dim_name, dist in pipeline.train_semantic_distribution.items():
        print(f"    [{{dim_name}}]")
        for cat, pct in dist.category_percentages.items():
            print(f"      - {{cat}}: {{pct:.1%}}")

# 保存结果
output_file = r"{parent_dir}\\tests\\attribution_results.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(attribution_results, f, ensure_ascii=False, indent=2, default=str)
print(f"\\n结果已保存: {{output_file}}")
'''.format(analysis_dir=analysis_service_dir, parent_dir=parent_dir)

    # 写入临时脚本
    temp_script = os.path.join(parent_dir, "tests", "_test_analysis.py")
    with open(temp_script, "w", encoding="utf-8") as f:
        f.write(test_script)

    # 运行
    result = subprocess.run([sys.executable, temp_script], capture_output=False)
    return result.returncode == 0


def main():
    """主测试流程"""
    print("=" * 70)
    print("服务逻辑完整测试")
    print("训练集: E:\\zzq\\训练集\\vehicle-13631-v18-cls9_split_80_10_10\\train")
    print("测试集: E:\\zzq\\误报")
    print("=" * 70)

    # 1. 测试 detection-service
    det_ok = test_detection_service()

    # 2. 测试 data-analysis-service
    analysis_ok = test_data_analysis_service()

    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()

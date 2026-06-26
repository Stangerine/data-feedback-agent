
import os
import sys
import time
import json

sys.path.insert(0, r"E:\zzq\agent_project\data-feedback-agent\services\data-analysis-service")

from config import get_config
from pipeline import AnalysisPipeline

cfg = get_config()
training_dir = cfg.get("data", {}).get("training_dir", "")
test_dir = cfg.get("data", {}).get("test_dir", "")

print(f"配置信息:")
print(f"  训练集: {training_dir}")
print(f"  测试集: {test_dir}")

# 使用已有的缓存目录
cache_dir = r"E:\zzq\agent_project\data-feedback-agent\tests\semantic_cache"
print(f"  缓存目录: {cache_dir}")

# 修改配置使用已有的缓存
cfg["semantic"]["cache_dir"] = cache_dir

# 初始化 Pipeline
print(f"\n初始化分析 Pipeline...")
pipeline = AnalysisPipeline(cfg)

print(f"预计算训练集分布...")
t0 = time.time()
pipeline.initialize()
init_time = time.time() - t0
print(f"初始化完成 ({init_time:.2f}s)")

# 获取测试图片
image_files = sorted([
    f for f in os.listdir(test_dir)
    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
])[:1]

print(f"\n测试图片: {len(image_files)} 张")

# 加载检测结果 (如果有)
det_results_file = r"E:\zzq\agent_project\data-feedback-agent\tests\detection_results.json"
detection_results = {}
if os.path.exists(det_results_file):
    with open(det_results_file, "r", encoding="utf-8") as f:
        det_results = json.load(f)
        for item in det_results:
            detection_results[item["image"]] = item

# 归因分析
attribution_results = []

for img_name in image_files:
    img_path = os.path.join(test_dir, img_name)
    print(f"\n{'─' * 70}")
    print(f"归因分析: {img_name}")
    print('─' * 70)

    # 从检测结果提取信息
    det_item = detection_results.get(img_name, {})
    detections = det_item.get("detections", [])
    verification = det_item.get("verification", {})

    false_positives = []
    false_negatives = []

    if verification.get("success"):
        data = verification["data"]
        false_positives = data.get("false_positives", [])
        false_negatives = data.get("missed_detections", [])

    print(f"  检测数: {len(detections)}")
    print(f"  误报数: {len(false_positives)}")
    print(f"  漏报数: {len(false_negatives)}")

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

        print(f"\n  [归因结果] ({analyze_time:.2f}s)")
        print(f"    归因类型: {result.get('attribution_type')}")
        print(f"    置信度: {result.get('confidence', 0):.2f}")
        print(f"    主因维度: {result.get('main_cause_dimension')}")
        print(f"    是否建议回流: {result.get('should_feedback')}")
        print(f"    回流建议: {result.get('feedback_suggestion', '')[:100]}...")

        print(f"\n    各维度归因:")
        for da in result.get('dimension_attributions', []):
            print(f"      - {da.get('dimension')}: {da.get('category')}")
            print(f"        训练集占比: {da.get('train_coverage', 0):.1%}")
            print(f"        覆盖缺口: {da.get('is_gap')}")

        attribution_results.append({
            "image": img_name,
            "attribution": result,
        })
    except Exception as e:
        print(f"  [ERROR] 归因失败: {e}")
        import traceback
        traceback.print_exc()

# 统计训练集分布
print(f"\n{'=' * 70}")
print("训练集分布统计")
print('=' * 70)

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

# 保存结果
output_file = r"E:\zzq\agent_project\data-feedback-agent\tests\attribution_results.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(attribution_results, f, ensure_ascii=False, indent=2, default=str)
print(f"\n结果已保存: {output_file}")

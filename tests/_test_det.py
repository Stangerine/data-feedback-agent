
import os
import sys
import time
import json

sys.path.insert(0, r"E:\zzq\agent_project\data-feedback-agent\services\detection-service")

from config import get_config
from detection import DetectionClient
from verifier import Verifier
from services.correction_service import CorrectionService

cfg = get_config()
test_dir = cfg.data.test_dir

print(f"配置信息:")
print(f"  检测 API: {cfg.detection.api_url}")
print(f"  LLM 模型: {cfg.llm.model}")
print(f"  测试目录: {test_dir}")

# 获取测试图片
image_files = sorted([
    f for f in os.listdir(test_dir)
    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
])[:3]

print(f"  测试图片: {len(image_files)} 张")

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
    print(f"\n{'─' * 70}")
    print(f"处理图片: {img_name}")
    print('─' * 70)

    # 步骤 1: 小模型检测
    print(f"\n  [步骤1] 小模型检测...", end=" ", flush=True)
    try:
        t0 = time.time()
        detections = det_client.detect(img_path, cfg.detection.box_threshold)
        detect_time = time.time() - t0
        print(f"OK ({detect_time:.2f}s)")
        print(f"    检测到 {len(detections)} 个目标:")
        for i, det in enumerate(detections, 1):
            print(f"      {i}. {det['class_name']} ({det['confidence']:.2f})")
    except Exception as e:
        print(f"FAIL: {e}")
        results.append({"image": img_name, "error": f"检测失败: {e}"})
        continue

    # 步骤 2: 大模型校验
    print(f"  [步骤2] 大模型校验...", end=" ", flush=True)
    try:
        t1 = time.time()
        verification = verifier.verify(img_path, detections)
        verify_time = time.time() - t1
        print(f"OK ({verify_time:.2f}s)")

        if verification["success"]:
            data = verification["data"]
            fps = data.get("false_positives", [])
            mds = data.get("missed_detections", [])
            overall = data.get("overall_assessment", {})
            print(f"    误报: {len(fps)} 个")
            for fp in fps:
                print(f"      - {fp.get('reported_class')} -> {fp.get('actual_class')}")
            print(f"    漏报: {len(mds)} 个")
            for md in mds:
                print(f"      - {md.get('actual_class_cn', md.get('actual_class'))}")
            print(f"    质量: {overall.get('detection_quality', '?')}")
        else:
            print(f"    校验失败: {verification.get('error')}")
    except Exception as e:
        print(f"FAIL: {e}")
        verification = {"success": False, "error": str(e)}

    # 步骤 3: 自动纠正
    print(f"  [步骤3] 自动纠正...", end=" ", flush=True)
    try:
        t2 = time.time()
        correction = correction_service.correct(img_path, detections, verification)
        correct_time = time.time() - t2
        print(f"OK ({correct_time:.2f}s)")

        artifacts = correction.get("artifacts", {})
        print(f"    输出目录: {artifacts.get('dir', '')}")
    except Exception as e:
        print(f"FAIL: {e}")
        correction = {"error": str(e)}

    results.append({
        "image": img_name,
        "detections": detections,
        "verification": verification,
        "correction": correction,
    })

# 保存结果
output_file = r"E:\zzq\agent_project\data-feedback-agent\tests\detection_results.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=str)
print(f"\n结果已保存: {output_file}")

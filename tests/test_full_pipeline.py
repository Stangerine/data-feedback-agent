#!/usr/bin/env python3
"""
完整流程测试：小模型检测 -> 大模型校验 -> 归因分析 -> 自动矫正

使用真实数据目录 E:\zzq\误报 进行测试
"""

import json
import os
import sys
import time
from pathlib import Path

# 添加路径 (从 tests/ 目录向上找到 services)
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(parent_dir, "services", "detection-service"))
sys.path.insert(0, os.path.join(parent_dir, "services", "data-analysis-service"))

def test_full_pipeline():
    """测试完整流程"""
    print("=" * 80)
    print("完整流程测试：小模型检测 -> 大模型校验 -> 归因分析 -> 自动矫正")
    print("=" * 80)

    # 测试图片目录
    test_dir = r"E:\zzq\误报"
    if not os.path.exists(test_dir):
        print(f"[ERROR] 测试目录不存在: {test_dir}")
        return

    # 获取测试图片
    image_files = sorted([
        f for f in os.listdir(test_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
    ])[:3]  # 只测试前3张

    if not image_files:
        print(f"[ERROR] 测试目录中没有图片: {test_dir}")
        return

    print(f"\n[INFO] 测试目录: {test_dir}")
    print(f"[INFO] 测试图片: {len(image_files)} 张")
    for img in image_files:
        print(f"   - {img}")

    # ── 步骤 1: 小模型检测 ─────────────────────────────────────
    print("\n" + "=" * 80)
    print("步骤 1: 小模型检测 (YOLO API)")
    print("=" * 80)

    from config import get_config
    from detection import DetectionClient

    cfg = get_config()
    print(f"检测 API: {cfg.detection.api_url}")
    print(f"置信度阈值: {cfg.detection.box_threshold}")

    det_client = DetectionClient(
        api_url=cfg.detection.api_url,
        model_id=cfg.detection.model_id,
        box_threshold=cfg.detection.box_threshold,
        timeout=cfg.detection.timeout,
    )

    detection_results = {}
    for img_name in image_files:
        img_path = os.path.join(test_dir, img_name)
        print(f"\n[DETECT] {img_name} ...", end=" ", flush=True)

        try:
            detections = det_client.detect(img_path, cfg.detection.box_threshold)
            detection_results[img_path] = detections
            print(f"[OK] 检测到 {len(detections)} 个目标")
            for i, det in enumerate(detections, 1):
                print(f"   {i}. {det['class_name']} ({det['confidence']:.2f}) bbox={det['bbox']}")
        except Exception as e:
            print(f"[FAIL] 检测失败: {e}")
            detection_results[img_path] = []

    # ── 步骤 2: 大模型校验 ─────────────────────────────────────
    print("\n" + "=" * 80)
    print("步骤 2: 大模型校验 (MiMo)")
    print("=" * 80)

    from verifier import Verifier

    print(f"LLM 模型: {cfg.llm.model}")
    print(f"LLM API: {cfg.llm.api_url}")

    verifier = Verifier()

    verification_results = {}
    for img_path, detections in detection_results.items():
        img_name = os.path.basename(img_path)
        print(f"\n[VERIFY] {img_name} ...", end=" ", flush=True)

        try:
            result = verifier.verify(img_path, detections)
            verification_results[img_path] = result

            if result["success"]:
                data = result["data"]
                fps = data.get("false_positives", [])
                mds = data.get("missed_detections", [])
                print(f"[OK] 成功")
                print(f"   误报: {len(fps)} 个")
                for fp in fps:
                    print(f"      - 第{fp.get('detection_index', '?')}条: {fp.get('reported_class', '?')} -> {fp.get('actual_class', '?')}")
                print(f"   漏报: {len(mds)} 个")
                for md in mds:
                    print(f"      - {md.get('actual_class_cn', md.get('actual_class', '?'))}: {md.get('location', '?')}")
                print(f"   耗时: {result['duration_ms']}ms")
            else:
                print(f"[FAIL] 失败: {result.get('error', '未知错误')}")
        except Exception as e:
            print(f"[FAIL] 校验异常: {e}")
            verification_results[img_path] = {"success": False, "error": str(e)}

    # ── 步骤 3: 归因分析 ───────────────────────────────────────
    print("\n" + "=" * 80)
    print("步骤 3: 归因分析 (多维度)")
    print("=" * 80)

    from config import get_config as get_analysis_config
    from pipeline import AnalysisPipeline

    analysis_cfg = get_analysis_config()
    print(f"训练集目录: {analysis_cfg.get('data', {}).get('training_dir', '未配置')}")

    pipeline = AnalysisPipeline(analysis_cfg)

    print("\n[INFO] 初始化训练集分布 (首次需要较长时间)...")
    pipeline.initialize()
    print("[OK] 初始化完成")

    attribution_results = {}
    for img_path, detections in detection_results.items():
        img_name = os.path.basename(img_path)
        print(f"\n[ATTRIBUTION] {img_name} ...", end=" ", flush=True)

        # 从校验结果提取误报和漏报
        verification = verification_results.get(img_path, {})
        false_positives = []
        false_negatives = []

        if verification.get("success"):
            data = verification["data"]
            false_positives = data.get("false_positives", [])
            false_negatives = data.get("missed_detections", [])

        try:
            result = pipeline.analyze_single(
                image_path=img_path,
                detections=detections,
                false_positives=false_positives,
                false_negatives=false_negatives,
                verification_result=verification,
            )
            attribution_results[img_path] = result

            print(f"[OK] 成功")
            print(f"   归因类型: {result.get('attribution_type')}")
            print(f"   置信度: {result.get('confidence', 0):.2f}")
            print(f"   主因维度: {result.get('main_cause_dimension')}")
            print(f"   是否建议回流: {result.get('should_feedback')}")
            print(f"   回流建议: {result.get('feedback_suggestion', '')[:100]}...")
        except Exception as e:
            print(f"[FAIL] 归因失败: {e}")
            import traceback
            traceback.print_exc()
            attribution_results[img_path] = {"error": str(e)}

    # ── 步骤 4: 自动矫正 ───────────────────────────────────────
    print("\n" + "=" * 80)
    print("步骤 4: 自动矫正 (大模型纠正)")
    print("=" * 80)

    from services.correction_service import CorrectionService

    correction_service = CorrectionService()

    correction_results = {}
    for img_path, detections in detection_results.items():
        img_name = os.path.basename(img_path)
        verification = verification_results.get(img_path, {})

        if not verification.get("success"):
            print(f"\n[SKIP] {img_name} (校验未成功)")
            continue

        print(f"\n[CORRECT] {img_name} ...", end=" ", flush=True)

        try:
            result = correction_service.correct(img_path, detections, verification)
            correction_results[img_path] = result

            print(f"[OK] 成功")
            artifacts = result.get("artifacts", {})
            print(f"   输出目录: {artifacts.get('dir', '')}")
            print(f"   小模型标注图: {artifacts.get('small_model_image', '')}")
            print(f"   大模型纠正图: {artifacts.get('corrected_image', '')}")
            print(f"   结果JSON: {artifacts.get('result_json', '')}")
        except Exception as e:
            print(f"[FAIL] 矫正失败: {e}")
            import traceback
            traceback.print_exc()
            correction_results[img_path] = {"error": str(e)}

    # ── 汇总报告 ───────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("汇总报告")
    print("=" * 80)

    print(f"\n📊 测试统计:")
    print(f"   测试图片数: {len(image_files)}")
    print(f"   检测成功: {sum(1 for d in detection_results.values() if d)} 张")
    print(f"   校验成功: {sum(1 for v in verification_results.values() if v.get('success'))} 张")
    print(f"   归因成功: {sum(1 for a in attribution_results.values() if 'error' not in a)} 张")
    print(f"   矫正成功: {sum(1 for c in correction_results.values() if 'error' not in c)} 张")

    # 统计误报和漏报
    total_fp = 0
    total_md = 0
    for v in verification_results.values():
        if v.get("success"):
            data = v["data"]
            total_fp += len(data.get("false_positives", []))
            total_md += len(data.get("missed_detections", []))

    print(f"\n📈 检测质量:")
    print(f"   总误报数: {total_fp}")
    print(f"   总漏报数: {total_md}")

    # 统计归因建议
    should_feedback = sum(1 for a in attribution_results.values() if a.get("should_feedback"))
    print(f"\n💡 回流建议:")
    print(f"   建议回流: {should_feedback} 张")
    print(f"   不建议回流: {len(attribution_results) - should_feedback} 张")

    # 保存详细结果
    output_file = os.path.join(test_dir, "test_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "test_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_images": image_files,
            "detection_results": {k: v for k, v in detection_results.items()},
            "verification_results": {k: v for k, v in verification_results.items()},
            "attribution_results": {k: v for k, v in attribution_results.items()},
            "correction_results": {k: v for k, v in correction_results.items()},
        }, f, ensure_ascii=False, indent=2)

    print(f"\n💾 详细结果已保存: {output_file}")
    print("\n" + "=" * 80)
    print("测试完成!")
    print("=" * 80)


if __name__ == "__main__":
    test_full_pipeline()

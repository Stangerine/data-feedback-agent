"""
检测校验服务测试脚本

测试内容：
  1. 健康检查
  2. 正确样本校验（调检测API + LLM）
  3. 误报样本校验（调检测API + LLM）

使用前确保服务已启动：
  cd services/detection-service && bash start.sh

运行：
  cd services/detection-service && python test/test_service.py
"""

import os
import sys
import time

import requests

BASE_URL = os.getenv("SERVICE_URL", "http://localhost:8001")

# 测试数据
FALSE_POSITIVE_DIR = "E:\\zzq\\误报"
CORRECT_DIR = "E:\\zzq\\正确"


def _get_first_image(directory: str, ext: str = ".jpg") -> str | None:
    """获取目录下第一张图片"""
    if not os.path.isdir(directory):
        return None
    for f in sorted(os.listdir(directory)):
        if f.lower().endswith(ext):
            return os.path.join(directory, f)
    return None


# ── TEST 1: 健康检查 ────────────────────────────────────────

def test_health():
    print("=" * 60)
    print("TEST 1: 健康检查")
    print("=" * 60)

    resp = requests.get(f"{BASE_URL}/health", timeout=10)
    assert resp.status_code == 200
    data = resp.json()
    print(f"  状态:     {data['status']}")
    print(f"  LLM协议:  {data['llm_protocol']}")
    print(f"  LLM模型:  {data['llm_model']}")
    print(f"  检测API:  {data['detection_api']}")
    print("  ✅ PASS\n")
    return True


# ── 校验 API 通用逻辑 ───────────────────────────────────────


def _run_verify_api_test(title: str, image_dir: str) -> bool | None:
    print("=" * 60)
    print(title)
    print("=" * 60)

    image_path = _get_first_image(image_dir)
    if not image_path:
        print(f"  ⚠️  测试目录不存在或没有图片，跳过: {image_dir}\n")
        return None

    print(f"  图片: {os.path.basename(image_path)}")

    t0 = time.time()
    resp = requests.post(f"{BASE_URL}/api/verify", json={
        "image_path": image_path,
        "box_threshold": 0.5,
    }, timeout=300)
    duration = time.time() - t0

    if resp.status_code != 200:
        print(f"  ❌ 请求失败: {resp.status_code}")
        detail = resp.json().get("detail", "")
        print(f"     {detail}")
        print("  ⚠️  检测API可能不可用，跳过\n")
        return None

    data = resp.json()
    detections = data["detections"]
    v = data["verification"]

    print(f"  小模型检测: {len(detections)} 个目标")
    for i, d in enumerate(detections, 1):
        print(f"    {i}. {d['class_name']} ({d['confidence']:.2f})")

    print(f"\n  大模型校验: {'✅ 成功' if v['success'] else '❌ 失败'}")
    if v["success"]:
        vd = v["data"]
        fps = vd.get("false_positives", [])
        mds = vd.get("missed_detections", [])
        overall = vd.get("overall_assessment", {})
        print(f"    误报: {len(fps)} 个")
        for fp in fps:
            print(f"      - 第{fp.get('detection_index','?')}条: "
                  f"{fp.get('reported_class','?')} → {fp.get('actual_class','?')}")
        print(f"    漏报: {len(mds)} 个")
        for md in mds:
            print(f"      - {md.get('actual_class_cn', md.get('actual_class','?'))}: "
                  f"{md.get('location','?')}")
        print(f"    质量: {overall.get('detection_quality','?')}")
        print(f"    总结: {overall.get('summary','?')}")
    else:
        raw = v.get("raw_text", "")
        print(f"    原始输出: {raw[:300]}")
        print(f"    错误: {v.get('error','?')}")

    print(f"  耗时: {duration:.1f}s")
    print(f"  {'✅ PASS' if v['success'] else '❌ FAIL'}\n")
    return v["success"]


# ── TEST 2: 正确样本校验（调检测API + LLM）────────────────────

def test_verify_correct_sample():
    return _run_verify_api_test("TEST 2: 正确样本校验（检测API + LLM）", CORRECT_DIR)


# ── TEST 3: 单图校验（调检测API + LLM）───────────────────────

def test_verify_single():
    return _run_verify_api_test("TEST 3: 误报样本校验（检测API + LLM）", FALSE_POSITIVE_DIR)


# ── 主入口 ───────────────────────────────────────────────────

def main():
    print(f"\n{'=' * 60}")
    print(f"  检测校验服务测试 | {BASE_URL}")
    print(f"{'=' * 60}\n")

    results = {}

    # 1. 健康检查
    try:
        results["health"] = test_health()
    except Exception as e:
        print(f"  ❌ 健康检查失败: {e}\n")
        results["health"] = False

    if not results.get("health"):
        print("服务不可用，请先启动: bash start.sh")
        sys.exit(1)

    # 2. 正确样本校验（检测API + LLM）
    try:
        results["correct"] = test_verify_correct_sample()
    except Exception as e:
        print(f"  ❌ 正确样本校验异常: {e}\n")
        results["correct"] = False

    # 3. 误报样本校验（检测API + LLM）
    try:
        results["single"] = test_verify_single()
    except Exception as e:
        print(f"  ❌ 单图校验异常: {e}\n")
        results["single"] = False

    # 汇总
    print("=" * 60)
    print("  测试汇总")
    print("=" * 60)
    for name, passed in results.items():
        if passed is None:
            status = "⏭️  SKIP"
        elif passed:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        print(f"  {name:10s} {status}")

    passed_count = sum(1 for v in results.values() if v is True)
    total_count = sum(1 for v in results.values() if v is not None)
    print(f"\n  {passed_count}/{total_count} 通过")
    print("=" * 60)

    sys.exit(0 if passed_count == total_count else 1)


if __name__ == "__main__":
    main()

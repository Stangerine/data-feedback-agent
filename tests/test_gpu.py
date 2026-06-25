#!/usr/bin/env python3
"""
测试 BGE-VL-large 模型是否在 GPU 上运行
"""

import os
import sys
import time

# 添加路径
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(parent_dir, "services", "data-analysis-service"))

def test_gpu():
    """测试 GPU 状态和模型加载"""
    print("=" * 60)
    print("BGE-VL-large GPU 测试")
    print("=" * 60)

    # 1. 检查 CUDA 是否可用
    print("\n[1] 检查 CUDA 环境...")
    try:
        import torch
        print(f"  PyTorch 版本: {torch.__version__}")
        print(f"  CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA 版本: {torch.version.cuda}")
            print(f"  GPU 数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"    显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
        else:
            print("  [WARNING] CUDA 不可用，将使用 CPU")
    except ImportError:
        print("  [ERROR] PyTorch 未安装")
        return

    # 2. 检查模型路径
    print("\n[2] 检查模型路径...")
    from config import get_config
    cfg = get_config()
    semantic_config = cfg.get("semantic", {})
    model_name = semantic_config.get("model_name", "")
    device = semantic_config.get("device", "cuda")

    print(f"  模型路径: {model_name}")
    print(f"  配置设备: {device}")

    if not os.path.exists(model_name):
        print(f"  [ERROR] 模型路径不存在: {model_name}")
        return

    # 3. 加载模型并测试
    print("\n[3] 加载 BGE-VL-large 模型...")
    try:
        from transformers import AutoModel, CLIPProcessor
        import torch

        t0 = time.time()

        # 加载模型
        print("  加载模型中...")
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

        # 移动到指定设备
        print(f"  移动模型到 {device}...")
        model = model.to(device)
        model.eval()

        # 加载 processor 并设置到模型
        print("  设置 processor...")
        model.set_processor(model_name)

        duration = time.time() - t0
        print(f"  模型加载完成，耗时: {duration:.2f}s")

        # 4. 检查模型所在设备
        print("\n[4] 检查模型设备...")
        model_device = next(model.parameters()).device
        print(f"  模型参数设备: {model_device}")

        if model_device.type == "cuda":
            print("  [OK] 模型在 GPU 上运行")
            # 显示显存使用
            print(f"  GPU 显存已用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"  GPU 显存缓存: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        else:
            print("  [WARNING] 模型在 CPU 上运行")

        # 5. 测试推理
        print("\n[5] 测试推理...")
        test_texts = ["挖掘机", "铲车", "大桩机"]

        with torch.no_grad():
            # 文本编码
            t1 = time.time()
            text_features = model.encode(text=test_texts)
            text_duration = time.time() - t1
            print(f"  文本编码耗时: {text_duration:.3f}s")
            print(f"  文本特征形状: {text_features.shape}")
            print(f"  文本特征设备: {text_features.device}")

        print("\n" + "=" * 60)
        print("测试完成!")
        print("=" * 60)

    except Exception as e:
        print(f"  [ERROR] 模型加载失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_gpu()

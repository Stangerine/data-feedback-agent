"""配置加载器 — 从全局配置文件读取配置"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml


_config: Optional[dict] = None


def _find_global_config() -> str:
    """查找全局配置文件"""
    # 从当前文件向上查找 config.yaml
    current = Path(__file__).resolve()
    for parent in current.parents:
        config_path = parent / "config.yaml"
        if config_path.exists():
            return str(config_path)

    # 默认路径 (Windows)
    return "E:\\zzq\\agent_project\\data-feedback-agent\\config.yaml"


def _apply_env_overrides(cfg: dict) -> dict:
    """环境变量覆盖"""
    env_map = {
        "DATA_ANALYSIS_PORT": ("server", "data_analysis_port"),
        "TRAINING_DIR": ("data", "training_dir"),
        "TEST_DIR": ("data", "test_dir"),
        "CLIP_MODEL": ("semantic", "model_name"),
        "CLIP_DEVICE": ("semantic", "device"),
        "CLIP_BATCH_SIZE": ("semantic", "batch_size"),
        "LLM_API_URL": ("llm", "api_url"),
        "LLM_API_KEY": ("llm", "api_key"),
        "LLM_MODEL": ("llm", "model"),
    }
    for env_key, path in env_map.items():
        val = os.environ.get(env_key)
        if val is not None:
            d = cfg
            for key in path[:-1]:
                d = d.setdefault(key, {})
            # 类型转换
            target = d.get(path[-1])
            if isinstance(target, int):
                d[path[-1]] = int(val)
            elif isinstance(target, float):
                d[path[-1]] = float(val)
            else:
                d[path[-1]] = val
    return cfg


def get_config() -> dict:
    """获取全局配置"""
    global _config
    if _config is not None:
        return _config

    config_path = _find_global_config()
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg = _apply_env_overrides(cfg)
    _config = cfg
    return _config


def get_llm_config() -> dict:
    """获取 LLM 配置（兼容旧格式）"""
    config = get_config()
    llm = config.get("llm", {})
    return {
        "protocol": "openai",
        "timeout": llm.get("timeout", 300),
        "temperature": llm.get("temperature", 0.1),
        "openai": {
            "api_url": llm.get("api_url", ""),
            "api_key": llm.get("api_key", ""),
            "model": llm.get("model", ""),
        }
    }

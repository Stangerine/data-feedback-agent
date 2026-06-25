"""
配置加载器 — 从全局配置文件加载配置
"""

import os
from pathlib import Path
from dataclasses import dataclass, field


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


def _load_yaml(path: str) -> dict:
    """加载 YAML 配置"""
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_config() -> dict:
    """加载配置文件，环境变量可覆盖关键字段"""
    config_path = _find_global_config()
    try:
        cfg = _load_yaml(config_path)
    except FileNotFoundError:
        raise RuntimeError(f"配置文件不存在: {config_path}")
    except Exception as e:
        raise RuntimeError(f"配置文件解析失败: {e}")

    # 环境变量覆盖
    env_map = {
        "DETECTION_API_URL": ("detection", "api_url"),
        "MODEL_ID": ("detection", "model_id"),
        "BOX_THRESHOLD": ("detection", "box_threshold"),
        "LLM_API_URL": ("llm", "api_url"),
        "LLM_API_KEY": ("llm", "api_key"),
        "LLM_MODEL": ("llm", "model"),
        "PORT": ("server", "detection_service_port"),
    }

    for env_key, path in env_map.items():
        val = os.getenv(env_key)
        if val is not None:
            obj = cfg
            for k in path[:-1]:
                obj = obj.setdefault(k, {})
            # 类型转换
            target = obj.get(path[-1])
            if isinstance(target, int):
                val = int(val)
            elif isinstance(target, float):
                val = float(val)
            obj[path[-1]] = val

    return cfg


# ── 配置数据类 ────────────────────────────────────────────────

@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    detection_service_port: int = 8001


@dataclass
class AuditConfig:
    enabled: bool = True
    dir: str = "./audit_logs"


@dataclass
class DetectionConfig:
    api_url: str = ""
    model_id: str = ""
    box_threshold: float = 0.5
    timeout: int = 60


@dataclass
class LLMConfig:
    api_url: str = ""
    api_key: str = ""
    model: str = ""
    timeout: int = 300
    temperature: float = 0.1


@dataclass
class AppConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)


def _dict_to_dataclass(cls, data: dict):
    """递归将 dict 转为 dataclass"""
    if not isinstance(data, dict):
        return data
    fields = {f.name: f for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    for k, v in data.items():
        if k in fields:
            ftype = fields[k].type
            # 处理嵌套 dataclass
            if hasattr(ftype, "__dataclass_fields__") and isinstance(v, dict):
                kwargs[k] = _dict_to_dataclass(ftype, v)
            else:
                kwargs[k] = v
    return cls(**kwargs)


def load_config() -> AppConfig:
    """加载并返回类型化的配置对象"""
    raw = _load_config()

    # 转换为 AppConfig 结构
    config_data = {
        "server": {
            "host": raw.get("server", {}).get("host", "0.0.0.0"),
            "detection_service_port": raw.get("server", {}).get("detection_service_port", 8001),
        },
        "detection": raw.get("detection", {}),
        "audit": raw.get("audit", {"enabled": True, "dir": "./audit_logs"}),
        "llm": raw.get("llm", {}),
    }

    return _dict_to_dataclass(AppConfig, config_data)


# 全局单例
_config: AppConfig | None = None


def get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config

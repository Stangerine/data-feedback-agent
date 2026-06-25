"""
LLM 客户端工厂 — 根据 protocol 创建对应客户端
"""

from .base import BaseLLMClient
from .openai_client import OpenAILLMClient
from .anthropic_client import AnthropicLLMClient
from .ollama_client import OllamaLLMClient


def create_llm_client(protocol: str, **kwargs) -> BaseLLMClient:
    """
    工厂函数：创建 LLM 客户端

    Args:
        protocol: openai | anthropic | ollama
        **kwargs: 传递给具体客户端的参数
    """
    clients = {
        "openai": OpenAILLMClient,
        "anthropic": AnthropicLLMClient,
        "ollama": OllamaLLMClient,
    }

    cls = clients.get(protocol)
    if cls is None:
        raise ValueError(f"不支持的 LLM 协议: {protocol}，可选: {list(clients.keys())}")

    return cls(**kwargs)


__all__ = ["BaseLLMClient", "create_llm_client"]

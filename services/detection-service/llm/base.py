"""
LLM 客户端基类 — 定义统一接口
"""

from abc import ABC, abstractmethod
from PIL import Image


class BaseLLMClient(ABC):
    """LLM 客户端抽象基类，所有协议实现必须继承"""

    def __init__(self, model: str, api_url: str, api_key: str = "",
                 timeout: int = 300, max_tokens: int = 2048, temperature: float = 0.1):
        self.model = model
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature

    @abstractmethod
    def chat(self, system_prompt: str, user_prompt: str,
             image: Image.Image | None = None,
             tools: list[dict] | None = None) -> dict:
        """
        发送多模态对话请求

        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            image: 可选的 PIL 图片

        Returns:
            dict: { success: bool, content: str, error: str|None }
        """
        ...

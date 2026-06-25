"""
Anthropic 兼容协议客户端 — 支持 Claude、MiMo(Anthropic接口) 等
"""

import base64
import io
import requests
from PIL import Image

from .base import BaseLLMClient


class AnthropicLLMClient(BaseLLMClient):
    """通过 Anthropic 兼容 API 调用多模态大模型"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.chat_url = f"{self.api_url}/v1/messages"
        print(f"[Anthropic] model={self.model}, url={self.chat_url}")

    def chat(self, system_prompt: str, user_prompt: str,
             image: Image.Image | None = None,
             tools: list[dict] | None = None) -> dict:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        # 构建 content
        content: list[dict] = []
        if image:
            img_b64 = self._image_to_base64(image)
            media_type = "image/jpeg"
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": img_b64,
                },
            })
        content.append({"type": "text", "text": user_prompt})

        payload = {
            "model": self.model,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": content},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        try:
            resp = requests.post(self.chat_url, headers=headers,
                                 json=payload, timeout=self.timeout)
            resp.raise_for_status()
            result = resp.json()

            # Anthropic 响应格式
            text = ""
            for block in result.get("content", []):
                if block.get("type") == "text":
                    text += block.get("text", "")

            if not text:
                return {"success": False, "content": "", "error": "模型返回空内容"}

            return {"success": True, "content": text, "tool_call": None, "error": None}

        except requests.exceptions.RequestException as e:
            return {"success": False, "content": "", "error": f"API 调用失败: {e}"}

    @staticmethod
    def _image_to_base64(image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=95)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

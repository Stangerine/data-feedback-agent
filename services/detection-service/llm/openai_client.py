"""
OpenAI 兼容协议客户端 — 支持 MiMo、GPT 等
"""

import base64
import io
import json
import requests
from PIL import Image

from .base import BaseLLMClient


class OpenAILLMClient(BaseLLMClient):
    """通过 OpenAI 兼容 API 调用多模态大模型"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.chat_url = f"{self.api_url}/chat/completions"
        print(f"[OpenAI] model={self.model}, url={self.chat_url}")

    def chat(self, system_prompt: str, user_prompt: str,
             image: Image.Image | None = None,
             tools: list[dict] | None = None) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # 构建 user content
        content: list[dict] = []
        if image:
            img_b64 = self._image_to_base64(image)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
            })
        content.append({"type": "text", "text": user_prompt})

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "temperature": self.temperature,
            "max_completion_tokens": self.max_tokens,
        }

        # function calling
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        try:
            resp = requests.post(self.chat_url, headers=headers,
                                 json=payload, timeout=self.timeout)
            resp.raise_for_status()
            result = resp.json()

            message = result.get("choices", [{}])[0].get("message", {})

            # 优先解析 tool_calls
            tool_calls = message.get("tool_calls")
            if tool_calls and len(tool_calls) > 0:
                try:
                    args = json.loads(tool_calls[0]["function"]["arguments"])
                    return {"success": True, "content": "", "tool_call": args, "error": None}
                except (json.JSONDecodeError, KeyError):
                    pass

            # fallback: 解析 content
            text = message.get("content", "")
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

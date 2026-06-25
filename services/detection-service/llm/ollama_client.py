"""
Ollama 本地部署客户端 — 支持 Ollama 本地服务
"""

import base64
import io
import requests
from PIL import Image

from .base import BaseLLMClient


class OllamaLLMClient(BaseLLMClient):
    """通过 Ollama 本地 API 调用多模态大模型"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.chat_url = f"{self.api_url}/chat/completions"
        print(f"[Ollama] model={self.model}, url={self.chat_url}")

    def chat(self, system_prompt: str, user_prompt: str,
             image: Image.Image | None = None,
             tools: list[dict] | None = None) -> dict:
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # 构建 user content — Ollama 格式
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
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        try:
            resp = requests.post(self.chat_url, headers=headers,
                                 json=payload, timeout=self.timeout)
            resp.raise_for_status()
            result = resp.json()

            # 兼容 Ollama 和 OpenAI 格式
            text = ""
            if "message" in result:
                # Ollama 格式
                text = result["message"].get("content", "")
            elif "choices" in result:
                # OpenAI 格式 (vLLM 等)
                text = result["choices"][0].get("message", {}).get("content", "")

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

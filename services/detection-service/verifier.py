"""
校验器 — 编排检测 + LLM 校验流程，通过 function calling 获取结构化输出
"""

import json
import re
import time
from datetime import datetime
from pathlib import Path

from PIL import Image
from pydantic import ValidationError

from config import get_config
from llm import create_llm_client
from llm.base import BaseLLMClient
from prompts.verification import SYSTEM_PROMPT, build_verify_prompt, build_verify_prompt_with_gt
from schemas import VerificationResult
from tools.function_calling import VERIFICATION_TOOL

MAX_ATTEMPTS = 2


class Verifier:
    """检测校验编排器"""

    def __init__(self, llm_client: BaseLLMClient | None = None):
        cfg = get_config()
        self._audit_cfg = cfg.audit
        if llm_client:
            self._llm = llm_client
        else:
            self._llm = create_llm_client(
                protocol="openai",
                model=cfg.llm.model,
                api_url=cfg.llm.api_url,
                api_key=cfg.llm.api_key,
                timeout=cfg.llm.timeout,
                temperature=cfg.llm.temperature,
            )

    def verify(self, image_path: str, detections: list[dict],
               ground_truth: list[dict] | None = None) -> dict:
        """
        对单张图片执行大模型校验

        优先通过 function calling 获取结构化输出，
        fallback 到 JSON 解析。
        """
        t0 = time.time()
        image = Image.open(image_path).convert("RGB")

        if ground_truth:
            user_prompt = build_verify_prompt_with_gt(detections, ground_truth)
        else:
            user_prompt = build_verify_prompt(detections)

        last_error = None
        raw_text = ""

        for attempt in range(1, MAX_ATTEMPTS + 1):
            current_prompt = user_prompt
            if attempt > 1 and last_error:
                current_prompt += (
                    f"\n\n[注意] 上次调用失败：{last_error}。"
                    f"请务必调用 report_verification 函数报告结果。"
                )

            resp = self._llm.chat(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=current_prompt,
                image=image,
                tools=[VERIFICATION_TOOL],
            )

            if not resp["success"]:
                last_error = resp.get("error", "模型返回空内容")
                continue

            # 1) 优先从 tool_calls 提取
            tool_call = resp.get("tool_call")
            if tool_call is not None:
                try:
                    result = VerificationResult(**tool_call)
                    audit_record = self._build_audit(
                        image_path, detections, ground_truth,
                        raw_text=json.dumps(tool_call, ensure_ascii=False),
                        result=result, attempt=attempt, t0=t0,
                        source="tool_call",
                        image_size={"width": image.width, "height": image.height},
                    )
                    self._save_audit(audit_record)
                    return self._build_response(audit_record)
                except ValidationError as e:
                    last_error = _validation_err(e)
                    continue

            # 2) fallback: 从 content 解析 JSON
            raw_text = resp.get("content", "")
            if raw_text:
                parsed = _parse_json(raw_text)
                if parsed:
                    try:
                        result = VerificationResult(**parsed)
                        audit_record = self._build_audit(
                            image_path, detections, ground_truth,
                            raw_text=raw_text,
                            result=result, attempt=attempt, t0=t0,
                            source="json_fallback",
                            image_size={"width": image.width, "height": image.height},
                        )
                        self._save_audit(audit_record)
                        return self._build_response(audit_record)
                    except ValidationError as e:
                        last_error = _validation_err(e)
                        continue

            last_error = last_error or "未返回有效结果"

        # 所有重试失败，也保存审计记录
        audit_record = self._build_audit(
            image_path, detections, ground_truth,
            raw_text=raw_text, result=None, attempt=MAX_ATTEMPTS, t0=t0,
            source="failed", error=last_error,
            image_size={"width": image.width, "height": image.height},
        )
        self._save_audit(audit_record)
        return self._build_response(audit_record)

    @staticmethod
    def _build_audit(image_path, detections, ground_truth,
                     raw_text, result, attempt, t0, source, error=None,
                     image_size=None):
        """构建审计记录"""
        cfg = get_config()
        return {
            "timestamp": datetime.now().isoformat(),
            "image_path": image_path,
            "image_size": image_size,
            "detections": detections,
            "ground_truth": ground_truth,
            "llm": {
                "protocol": cfg.llm.protocol,
                "model": cfg.llm.model,
            },
            "source": source,
            "raw_text": raw_text,
            "result": result.model_dump() if result else None,
            "success": result is not None,
            "error": error,
            "attempts": attempt,
            "duration_ms": int((time.time() - t0) * 1000),
        }

    @staticmethod
    def _build_response(audit_record):
        """从审计记录构建 API 响应"""
        return {
            "success": audit_record["success"],
            "data": audit_record["result"],
            "raw_text": audit_record["raw_text"],
            "error": audit_record["error"],
            "duration_ms": audit_record["duration_ms"],
            "image_path": audit_record["image_path"],
            "image_size": audit_record.get("image_size"),
            "detection_count": len(audit_record["detections"]),
            "attempts": audit_record["attempts"],
        }

    def _save_audit(self, record: dict):
        """保存审计记录到文件"""
        if not self._audit_cfg.enabled:
            return

        audit_dir = Path(self._audit_cfg.dir)
        audit_dir.mkdir(parents=True, exist_ok=True)

        # 按日期分目录
        date_str = datetime.now().strftime("%Y-%m-%d")
        date_dir = audit_dir / date_str
        date_dir.mkdir(exist_ok=True)

        # 文件名: 时间戳 + 图片名
        img_name = Path(record["image_path"]).stem
        ts = datetime.now().strftime("%H%M%S-%f")
        filename = f"{ts}_{img_name}.json"

        filepath = date_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)


def _validation_err(e: ValidationError) -> str:
    errors = e.errors()
    return f"字段校验失败: {errors[0]['msg']}" if errors else str(e)


def _parse_json(text: str) -> dict | None:
    """从 LLM 输出中提取 JSON"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None

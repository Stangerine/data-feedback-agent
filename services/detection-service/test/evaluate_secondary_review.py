"""Evaluate GPT secondary review against labeled image directories.

This script calls the production `/api/verify` endpoint, so detections come
from the configured small-model API before multimodal LLM review.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(directory: str, limit: int | None) -> list[Path]:
    paths = sorted(
        p
        for p in Path(directory).rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    return paths[:limit] if limit is not None else paths


def review_image(service_url: str, image_path: Path, timeout: int) -> dict[str, Any]:
    started = time.time()
    response = requests.post(
        f"{service_url.rstrip('/')}/api/verify",
        json={"image_path": str(image_path), "box_threshold": 0.5},
        timeout=timeout,
    )
    duration_s = round(time.time() - started, 2)
    response.raise_for_status()
    data = response.json()
    data["duration_s"] = duration_s
    return data


def classify_review(data: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    verification = data.get("verification") or {}
    if not verification.get("success"):
        return "failed", {
            "fp_count": None,
            "md_count": None,
            "quality": "failed",
            "summary": verification.get("error") or "verification failed",
        }

    result = verification.get("data") or {}
    fps = result.get("false_positives") or []
    mds = result.get("missed_detections") or []
    overall = result.get("overall_assessment") or {}
    predicted = "clean" if len(fps) == 0 and len(mds) == 0 else "problematic"
    return predicted, {
        "fp_count": len(fps),
        "md_count": len(mds),
        "quality": overall.get("detection_quality", ""),
        "summary": overall.get("summary", ""),
        "false_positives": fps,
        "missed_detections": mds,
    }


def evaluate_group(
    service_url: str,
    label: str,
    expected: str,
    images: list[Path],
    timeout: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, image_path in enumerate(images, 1):
        print(f"[{label} {idx}/{len(images)}] {image_path.name}", flush=True)
        row: dict[str, Any] = {
            "label": label,
            "expected": expected,
            "image_path": str(image_path),
            "image_name": image_path.name,
        }
        try:
            data = review_image(service_url, image_path, timeout)
            predicted, details = classify_review(data)
            row.update(details)
            row["predicted"] = predicted
            row["match"] = predicted == expected
            row["duration_s"] = data["duration_s"]
            row["detection_count"] = len(data.get("detections") or [])
            row["detections"] = data.get("detections") or []
            row["audit_image_path"] = (data.get("verification") or {}).get("image_path")
        except Exception as exc:
            row.update(
                {
                    "predicted": "failed",
                    "match": False,
                    "error": str(exc),
                    "duration_s": None,
                    "detection_count": None,
                }
            )
        rows.append(row)
    return rows


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_label: dict[str, Any] = {}
    for label in sorted({row["label"] for row in rows}):
        label_rows = [row for row in rows if row["label"] == label]
        by_label[label] = {
            "total": len(label_rows),
            "matched": sum(1 for row in label_rows if row.get("match")),
            "failed": sum(1 for row in label_rows if row.get("predicted") == "failed"),
            "predicted": dict(Counter(row.get("predicted") for row in label_rows)),
            "avg_duration_s": round(
                sum(row.get("duration_s") or 0 for row in label_rows)
                / max(1, sum(1 for row in label_rows if row.get("duration_s") is not None)),
                2,
            ),
        }

    return {
        "total": len(rows),
        "matched": sum(1 for row in rows if row.get("match")),
        "failed": sum(1 for row in rows if row.get("predicted") == "failed"),
        "accuracy": round(
            sum(1 for row in rows if row.get("match")) / len(rows), 4
        )
        if rows
        else 0,
        "by_label": by_label,
    }


def write_markdown(output_md: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# GPT-5.5 Secondary Review Evaluation",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- Total: {summary['total']}",
        f"- Matched: {summary['matched']}",
        f"- Failed: {summary['failed']}",
        f"- Accuracy: {summary['accuracy']:.2%}",
        "",
        "## By Label",
        "",
        "| Label | Total | Matched | Failed | Predicted | Avg Duration(s) |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for label, item in summary["by_label"].items():
        lines.append(
            f"| {label} | {item['total']} | {item['matched']} | {item['failed']} | "
            f"{json.dumps(item['predicted'], ensure_ascii=False)} | {item['avg_duration_s']} |"
        )

    lines.extend(
        [
            "",
            "## Per Image",
            "",
            "| Label | Image | Expected | Predicted | Match | Detections | FP | MD | Quality | Summary |",
            "|---|---|---|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in rows:
        summary_text = str(row.get("summary") or row.get("error") or "").replace("|", "\\|")
        lines.append(
            f"| {row['label']} | {row['image_name']} | {row['expected']} | "
            f"{row.get('predicted')} | {row.get('match')} | {row.get('detection_count')} | "
            f"{row.get('fp_count')} | {row.get('md_count')} | {row.get('quality', '')} | "
            f"{summary_text} |"
        )

    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--service-url", default="http://127.0.0.1:8001")
    parser.add_argument("--correct-dir", default="E:\\zzq\\正确")
    parser.add_argument("--wrong-dir", default="E:\\zzq\\误报")
    parser.add_argument("--correct-limit", type=int, default=None)
    parser.add_argument("--wrong-limit", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--output-dir",
        default="E:\\zzq\\agent_project\\data-feedback-agent\\services\\detection-service\\eval_results",
    )
    args = parser.parse_args()

    correct_images = collect_images(args.correct_dir, args.correct_limit)
    wrong_images = collect_images(args.wrong_dir, args.wrong_limit)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    rows.extend(
        evaluate_group(args.service_url, "正确", "clean", correct_images, args.timeout)
    )
    rows.extend(
        evaluate_group(args.service_url, "误报", "problematic", wrong_images, args.timeout)
    )

    summary = build_summary(rows)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json = output_dir / f"secondary_review_eval_{timestamp}.json"
    output_md = output_dir / f"secondary_review_eval_{timestamp}.md"
    output_json.write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(output_md, summary, rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"JSON: {output_json}")
    print(f"Markdown: {output_md}")


if __name__ == "__main__":
    main()

"""语义维度分析器基类 — 使用 CLIP 进行语义匹配"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from schemas import (
    DatasetSemanticDistribution,
    SemanticDimensionResult,
)

logger = logging.getLogger(__name__)


class BaseSemanticAnalyzer:
    """语义维度分析器基类

    使用 CLIP 模型计算图片与语义提示词的相似度，
    对特定维度进行分类分析。
    """

    # 子类需要定义
    DIMENSION_NAME: str = ""
    CATEGORIES: Dict[str, str] = {}  # category -> prompt

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.processor = None
        self._load_failed = False
        self._text_embeddings: Optional[Dict[str, np.ndarray]] = None

        # 配置参数
        self.model_name = config.get("model_name", "E:\\zzq\\model\\BGE-VL-large")
        self.device = config.get("device", "cuda:0")
        self.batch_size = config.get("batch_size", 16)
        self.cache_dir = config.get("cache_dir", "./semantic_cache")

    def _load_model(self) -> bool:
        """懒加载 BGE-VL-large 模型"""
        if self.model is not None:
            return True
        if self._load_failed:
            return False

        try:
            import sys
            import torch
            from transformers import AutoModel, AutoProcessor, CLIPProcessor

            # 添加模型目录到路径，以便加载自定义模型类
            model_dir = str(self.model_name)
            if model_dir not in sys.path:
                sys.path.insert(0, model_dir)

            logger.info(f"[{self.DIMENSION_NAME}] 加载 BGE-VL-large 模型: {self.model_name}")

            # 使用 AutoModel 加载自定义模型（config.json 中定义了 auto_map）
            self.model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,  # CPU 模式使用 float32
                trust_remote_code=True,
            ).to(self.device)
            self.model.eval()

            # 加载 processor
            self.processor = CLIPProcessor.from_pretrained(self.model_name)

            # 设置 processor 到模型中（BGE-VL-large 的自定义方法需要）
            self.model.set_processor(self.model_name)

            logger.info(f"[{self.DIMENSION_NAME}] BGE-VL-large 模型加载成功 (device={self.device})")
            return True
        except Exception as e:
            logger.error(f"[{self.DIMENSION_NAME}] BGE-VL-large 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            self._load_failed = True
            return False

    def _compute_text_embeddings(self) -> Dict[str, np.ndarray]:
        """计算该维度所有语义提示词的文本 embedding（带缓存）"""
        if self._text_embeddings is not None:
            return self._text_embeddings

        if not self._load_model():
            return {}

        import torch

        try:
            categories = list(self.CATEGORIES.keys())
            texts = [self.CATEGORIES[cat] for cat in categories]

            logger.info(f"[{self.DIMENSION_NAME}] 计算 {len(texts)} 个类别的文本 embedding...")

            # 使用 BGE-VL-large 的 encode 方法
            with torch.no_grad():
                text_features = self.model.encode(text=texts)

            # 确保是 2D 数组
            if text_features.dim() == 1:
                text_features = text_features.unsqueeze(0)

            text_embeddings = text_features.cpu().float().numpy()
            self._text_embeddings = {
                "categories": categories,
                "embeddings": text_embeddings,
            }

            logger.info(f"[{self.DIMENSION_NAME}] 文本 embedding 计算完成: shape={text_embeddings.shape}")
            return self._text_embeddings
        except Exception as e:
            logger.error(f"[{self.DIMENSION_NAME}] 计算文本 embedding 失败: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _compute_image_embeddings(self, image_paths: List[str]) -> Tuple[np.ndarray, List[str]]:
        """计算图片的 embedding

        Returns:
            (embeddings, valid_paths): embeddings 数组和对应的有效路径列表
        """
        if not self._load_model():
            return np.array([]), []

        import torch

        # 过滤不存在的图片
        valid_paths = [p for p in image_paths if Path(p).exists()]
        if not valid_paths:
            return np.array([]), []

        # 检查缓存
        cache_key = hashlib.md5(
            (self.DIMENSION_NAME + json.dumps(sorted(valid_paths))).encode()
        ).hexdigest()
        cache_path = Path(self.cache_dir)
        cache_file = cache_path / f"{cache_key}.npy"
        cache_meta = cache_path / f"{cache_key}_paths.json"

        if cache_file.exists() and cache_meta.exists():
            try:
                cached_paths = json.loads(cache_meta.read_text())
                if cached_paths == valid_paths:
                    embeddings = np.load(str(cache_file))
                    logger.info(f"[{self.DIMENSION_NAME}] 缓存命中: {len(valid_paths)} 张图片")
                    return embeddings, valid_paths
            except Exception:
                pass

        # 批量计算 embedding
        all_embeddings = []
        for i in range(0, len(valid_paths), self.batch_size):
            batch_paths = valid_paths[i:i + self.batch_size]
            try:
                logger.info(f"[{self.DIMENSION_NAME}] 处理图片批次 {i // self.batch_size + 1}: {len(batch_paths)} 张")

                # 使用 BGE-VL-large 的 encode 方法
                with torch.no_grad():
                    image_features = self.model.encode(images=batch_paths)

                # 确保是 2D 数组
                if image_features.dim() == 1:
                    image_features = image_features.unsqueeze(0)

                all_embeddings.append(image_features.cpu().float().numpy())
            except Exception as e:
                logger.warning(f"[{self.DIMENSION_NAME}] 批次 {i // self.batch_size} 处理失败: {e}")
                import traceback
                traceback.print_exc()
                continue

        if not all_embeddings:
            return np.array([]), []

        embeddings = np.concatenate(all_embeddings, axis=0)

        # 保存缓存
        try:
            cache_path.mkdir(parents=True, exist_ok=True)
            np.save(str(cache_file), embeddings)
            cache_meta.write_text(json.dumps(valid_paths))
            logger.info(f"[{self.DIMENSION_NAME}] 缓存已保存: {len(valid_paths)} 张图片")
        except Exception as e:
            logger.warning(f"[{self.DIMENSION_NAME}] 保存缓存失败: {e}")

        return embeddings, valid_paths

    def analyze_single(self, image_path: str) -> SemanticDimensionResult:
        """分析单张图片的该维度"""
        result = SemanticDimensionResult(
            dimension=self.DIMENSION_NAME,
            best_category="unknown",
            confidence=0.0,
        )

        if not Path(image_path).exists():
            logger.warning(f"[{self.DIMENSION_NAME}] 图片不存在: {image_path}")
            return result

        # 计算文本 embedding
        text_embeddings = self._compute_text_embeddings()
        if not text_embeddings:
            return result

        # 计算图片 embedding
        image_embedding, _ = self._compute_image_embeddings([image_path])
        if len(image_embedding) == 0:
            return result

        image_emb = image_embedding[0]  # (D,)
        categories = text_embeddings["categories"]
        text_embs = text_embeddings["embeddings"]  # (N_categories, D)

        # 计算余弦相似度
        similarities = np.dot(text_embs, image_emb)
        # softmax 归一化得到置信度
        exp_sim = np.exp(similarities - np.max(similarities))
        confidence_scores = exp_sim / exp_sim.sum()

        best_idx = int(np.argmax(similarities))
        best_category = categories[best_idx]
        confidence = float(confidence_scores[best_idx])

        # 构建相似度分布
        sim_distribution = {
            cat: float(sim)
            for cat, sim in zip(categories, similarities)
        }

        return SemanticDimensionResult(
            dimension=self.DIMENSION_NAME,
            best_category=best_category,
            confidence=confidence,
            similarities=sim_distribution,
        )

    def analyze_batch(self, image_paths: List[str]) -> List[SemanticDimensionResult]:
        """批量分析图片的该维度"""
        if not image_paths:
            return []

        # 计算文本 embedding
        text_embeddings = self._compute_text_embeddings()
        if not text_embeddings:
            return [SemanticDimensionResult(
                dimension=self.DIMENSION_NAME,
                best_category="unknown",
                confidence=0.0,
            ) for _ in image_paths]

        # 计算图片 embedding
        image_embeddings, valid_paths = self._compute_image_embeddings(image_paths)
        if len(image_embeddings) == 0:
            return [SemanticDimensionResult(
                dimension=self.DIMENSION_NAME,
                best_category="unknown",
                confidence=0.0,
            ) for _ in image_paths]

        # 建立路径到 embedding 的映射
        path_to_emb = {path: emb for path, emb in zip(valid_paths, image_embeddings)}

        categories = text_embeddings["categories"]
        text_embs = text_embeddings["embeddings"]

        results = []
        for image_path in image_paths:
            if image_path not in path_to_emb:
                results.append(SemanticDimensionResult(
                    dimension=self.DIMENSION_NAME,
                    best_category="unknown",
                    confidence=0.0,
                ))
                continue

            image_emb = path_to_emb[image_path]

            # 计算余弦相似度
            similarities = np.dot(text_embs, image_emb)
            # softmax 归一化得到置信度
            exp_sim = np.exp(similarities - np.max(similarities))
            confidence_scores = exp_sim / exp_sim.sum()

            best_idx = int(np.argmax(similarities))
            best_category = categories[best_idx]
            confidence = float(confidence_scores[best_idx])

            # 构建相似度分布
            sim_distribution = {
                cat: float(sim)
                for cat, sim in zip(categories, similarities)
            }

            results.append(SemanticDimensionResult(
                dimension=self.DIMENSION_NAME,
                best_category=best_category,
                confidence=confidence,
                similarities=sim_distribution,
            ))

        return results

    def compute_distribution(
        self,
        results: List[SemanticDimensionResult]
    ) -> DatasetSemanticDistribution:
        """计算数据集的该维度分布统计"""
        category_counts: Dict[str, int] = {}
        total = 0

        for result in results:
            if result.best_category != "unknown":
                category_counts[result.best_category] = category_counts.get(result.best_category, 0) + 1
                total += 1

        # 计算百分比
        category_percentages = {}
        if total > 0:
            for cat, count in category_counts.items():
                category_percentages[cat] = count / total

        return DatasetSemanticDistribution(
            dimension=self.DIMENSION_NAME,
            category_counts=category_counts,
            category_percentages=category_percentages,
        )

"""维度 1: 类别覆盖分析 — 对比训练集与测试集的类别分布差异"""

from __future__ import annotations

from typing import Dict, List

from schemas import ClassDistribution


class ClassAnalyzer:
    """类别覆盖分析器"""

    def __init__(self, config: dict):
        pass

    def analyze(
        self,
        train_dist: List[ClassDistribution],
        test_dist: List[ClassDistribution],
    ) -> dict:
        """
        对比训练集和测试集的类别分布。

        返回:
            - train_class_map: 训练集类别计数 {class_id: count}
            - test_class_map: 测试集类别计数
            - coverage_gaps: 训练集覆盖薄弱的类别
            - overrepresented_in_test: 测试集中占比异常高的类别
            - class_scores: 每个测试类别的回流价值分 (0-1)
        """
        train_map = {cd.class_id: cd.count for cd in train_dist}
        test_map = {cd.class_id: cd.count for cd in test_dist}
        train_total = sum(train_map.values()) or 1
        test_total = sum(test_map.values()) or 1

        # 计算训练集各类别占比
        train_pcts = {cid: cnt / train_total for cid, cnt in train_map.items()}
        test_pcts = {cid: cnt / test_total for cid, cnt in test_map.items()}

        # 识别训练集覆盖薄弱类别 (占比 < 2%)
        coverage_gaps = []
        for cd in train_dist:
            if train_pcts.get(cd.class_id, 0) < 0.02:
                coverage_gaps.append({
                    "class_id": cd.class_id,
                    "class_name": cd.class_name,
                    "train_pct": round(train_pcts.get(cd.class_id, 0) * 100, 2),
                })

        # 识别测试集中占比异常高的类别 (测试占比 > 训练占比 * 1.5)
        overrepresented = []
        for cd in test_dist:
            test_pct = test_pcts.get(cd.class_id, 0)
            train_pct = train_pcts.get(cd.class_id, 0)
            if train_pct > 0 and test_pct > train_pct * 1.5:
                overrepresented.append({
                    "class_id": cd.class_id,
                    "class_name": cd.class_name,
                    "test_pct": round(test_pct * 100, 2),
                    "train_pct": round(train_pct * 100, 2),
                    "ratio": round(test_pct / train_pct, 2),
                })
            elif train_pct == 0 and test_pct > 0:
                overrepresented.append({
                    "class_id": cd.class_id,
                    "class_name": cd.class_name,
                    "test_pct": round(test_pct * 100, 2),
                    "train_pct": 0,
                    "ratio": float("inf"),
                })

        # 计算每个测试类别的回流价值分
        class_scores = {}
        for cd in test_dist:
            cid = cd.class_id
            train_pct = train_pcts.get(cid, 0)
            test_pct = test_pcts.get(cid, 0)

            # 稀有类回流价值高
            rarity_score = max(0, 1 - train_pct * 10)  # 训练占比越低，分数越高

            # 测试集中该类占比越高，说明模型在该类问题越大
            prevalence_score = min(1, test_pct * 5)

            # 综合
            score = 0.6 * rarity_score + 0.4 * prevalence_score
            class_scores[cid] = round(score, 4)

        return {
            "train_class_map": train_map,
            "test_class_map": test_map,
            "coverage_gaps": coverage_gaps,
            "overrepresented_in_test": overrepresented,
            "class_scores": class_scores,
        }

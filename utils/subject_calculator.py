# -*- coding: utf-8 -*-
"""
SubjectCalculator: 多维学科综合计算类（支持多策略）
-------------------------------------------------
策略：
 - default：强度 intensity + 广度 breadth + 平衡性 balance 综合打分
 - weighted：五维外部权重融合（各维内部归一化 + 融合排序）
-------------------------------------------------
"""

import os
import json
import math
import numpy as np
from collections import defaultdict
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()


class SubjectCalculator:
    """多维学科综合计算器"""

    DEFAULT_WEIGHTS = {
        "title_abs": float(os.getenv("WEIGHT_TITLE_ABS", 0.05)),
        "incites": float(os.getenv("WEIGHT_INCITES", 0.8)),
        "refs": float(os.getenv("WEIGHT_REFS", 0.05)),
        "openalex": float(os.getenv("WEIGHT_OPENALEX", 0.05)),
        "author": float(os.getenv("WEIGHT_AUTHOR", 0.05)),
    }

    def __init__(
        self,
        view_configs: Dict[str, Dict[str, Any]] = None,
        alpha: float = 0.3,
        beta: float = 0.2,
        rel_thr: float = 0.6,
        cover_thr: float = 0.8,
        topk_cross: int = 3,
        strategy: str = "default",
        debug: bool = False,
    ):
        """
        初始化计算器
        - alpha, beta：调节广度和平衡性的影响
        - rel_thr, cover_thr, topk_cross：交叉筛选控制参数
        - strategy：计算策略（default / weighted）
        """
        self.alpha = alpha
        self.beta = beta
        self.rel_thr = rel_thr
        self.cover_thr = cover_thr
        self.topk_cross = topk_cross
        self.strategy = strategy
        self.debug = debug

        self.view_configs = view_configs or {
            "title_abs": {"col": "list_title_abs", "weight": self.DEFAULT_WEIGHTS["title_abs"]},
            "incites": {"col": "list_incites_direction", "weight": self.DEFAULT_WEIGHTS["incites"]},
            "refs": {"col": "list_ref", "weight": self.DEFAULT_WEIGHTS["refs"]},
            "openalex": {"col": "list_openalex", "weight": self.DEFAULT_WEIGHTS["openalex"]},
            "author": {"col": "list_author_aff_qwen", "weight": self.DEFAULT_WEIGHTS["author"]},
        }

    # ==========================================================
    # ------------------ 通用辅助函数 ---------------------------
    # ==========================================================

    @staticmethod
    def _safe_eval_list(x):
        """安全解析字符串为列表"""
        try:
            val = eval(x) if isinstance(x, str) else x
            return val if isinstance(val, list) else []
        except Exception:
            return []

    @staticmethod
    def _dict_from_list(lst):
        """扁平列表转 dict"""
        out = {}
        if not isinstance(lst, list):
            return out
        for i in lst:
            if isinstance(i, (list, tuple)) and len(i) == 2:
                out[i[0]] = float(i[1])
        return out

    @staticmethod
    def _flatten_nested(lst):
        """展开多层嵌套结构（作者机构）"""
        out = []
        for i in lst:
            if isinstance(i, (list, tuple)) and len(i) == 2 and isinstance(i[0], str):
                out.append(i)
            elif isinstance(i, (list, tuple)):
                out.extend(SubjectCalculator._flatten_nested(i))
        return out

    # ==========================================================
    # ------------------ 主调度入口 ----------------------------
    # ==========================================================
    def calc(self, row: dict):
        """根据策略选择计算方法"""
        if self.strategy == "weighted":
            return self._calc_weighted(row)
        else:
            return self._calc_default(row)

    # ==========================================================
    # ------------------ 策略 1：default ------------------------
    # ==========================================================
    def _calc_default(self, row: dict):
        """默认策略：intensity + breadth + balance 综合打分"""
        # 1️⃣ 解析视角
        views = []
        for name, cfg in self.view_configs.items():
            col = cfg["col"]
            weight = cfg.get("weight", 0.1)
            raw_list = self._safe_eval_list(row.get(col, "[]"))
            # 若为二元组格式 [(学科, 分数)]
            vals = self._dict_from_list(raw_list)
            views.append((name, vals, weight))

        subjects = set().union(*[set(v.keys()) for _, v, _ in views])
        if not subjects:
            return {"primary": None, "cross": [], "detail": {}}

        # 2️⃣ 计算贡献矩阵
        contrib, support = defaultdict(dict), defaultdict(set)
        for name, view_tf, weight in views:
            for d, tf in view_tf.items():
                contrib[d][name] = weight * float(tf)
                support[d].add(name)

        # 3️⃣ 计算 intensity / breadth / balance
        scores, detail = {}, {}
        for d in subjects:
            cs = contrib[d]
            intensity = sum(cs.values())
            breadth = len(cs) / len(views)
            if len(cs) >= 2:
                gm = math.prod(cs.values()) ** (1 / len(cs))
                balance = gm / max(cs.values())
            else:
                balance = 0.0
            score = intensity * (1 + self.alpha * breadth) * (1 + self.beta * balance)
            scores[d] = score
            detail[d] = {
                "intensity": intensity,
                "breadth": breadth,
                "balance": balance,
                "views": sorted(list(support[d])),
            }

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        primary, primary_val = ranked[0]

        # 交叉学科筛选
        def tool_like(d):
            s = support[d]
            return ("title_abs" not in s) and ("incites" not in s)

        cross, cum = [], primary_val
        total = sum(scores.values())
        for d, v in ranked[1:]:
            if v < self.rel_thr * primary_val:
                continue
            if len(support[d]) < 2:
                continue
            if tool_like(d):
                continue
            cross.append(d)
            cum += v
            if cum >= self.cover_thr * total or len(cross) >= self.topk_cross:
                break

        return {"primary": primary, "cross": cross, "detail": detail}

    # ==========================================================
    # ------------------ 策略 2：weighted -----------------------
    # ==========================================================
    def _calc_weighted(self, row: dict):
        """
        新策略：五维外部权重融合（各维内部归一化 + 加权聚合）
        ✅ 支持混合归一化策略：
            - title_abs: softmax
            - incites: minmax
            - refs: softmax
            - openalex: minmax
            - author: minmax
        ✅ 自动补全 detail 维度（缺省填 0）
        """
        import numpy as np
        from collections import defaultdict

        # 1️⃣ 解析每个维度数据
        dims = {}
        for name, cfg in self.view_configs.items():
            col = cfg["col"]
            raw = self._safe_eval_list(row.get(col, "[]"))
            flat = self._flatten_nested(raw)
            dims[name] = flat

        # 2️⃣ 混合归一化策略配置
        norm_map = {
            "title_abs": "softmax",
            "incites": "softmax",
            "refs": "softmax",
            "openalex": "softmax",
            "author": "softmax",
        }

        def normalize(pairs, method="minmax"):
            """对单个维度内部进行归一化"""
            if not pairs:
                return {}
            subj2score = defaultdict(float)
            for subj, val in pairs:
                subj2score[subj] += float(val)
            vals = np.array(list(subj2score.values()), dtype=float)

            # softmax归一化
            if method == "softmax":
                e_x = np.exp(vals - np.max(vals))
                probs = e_x / e_x.sum() if e_x.sum() != 0 else np.ones_like(e_x) / len(e_x)
                return {k: float(v) for k, v in zip(subj2score.keys(), probs)}

            # min-max归一化
            lo, hi = vals.min(), vals.max()
            if hi == lo:
                return {k: 1.0 for k in subj2score}
            return {k: (v - lo) / (hi - lo) for k, v in subj2score.items()}

        # 3️⃣ 各维度归一化后得分表
        dim_scores = {
            dim: normalize(pairs, norm_map.get(dim, "minmax"))
            for dim, pairs in dims.items()
        }

        # 4️⃣ 按外部权重融合得分
        total_scores = defaultdict(float)
        detail = defaultdict(lambda: {"total": 0.0, "views": {}})

        for dim, subj_scores in dim_scores.items():
            w = self.view_configs[dim].get("weight", 0.0)
            for subj, val in subj_scores.items():
                total_scores[subj] += w * val
                detail[subj]["total"] += w * val
                detail[subj]["views"][dim] = val

        if not total_scores:
            return {"primary": None, "cross": [], "detail": {}}

        # 5️⃣ 排序选主交叉
        ranked = sorted(total_scores.items(), key=lambda kv: kv[1], reverse=True)
        primary, _ = ranked[0]
        cross = [d for d, _ in ranked[1:self.topk_cross + 1]]

        # ✅ 自动补全所有 subject 的 5 个维度（缺省填 0）
        all_dims = list(self.view_configs.keys())
        for subj in detail:
            for dim in all_dims:
                if dim not in detail[subj]["views"]:
                    detail[subj]["views"][dim] = 0.0

        # ✅ Debug 输出（可选）
        if self.debug:
            print(f"\n🧩 综合加权结果：共 {len(detail)} 个学科")
            for subj, info in sorted(detail.items(), key=lambda kv: kv[1]["total"], reverse=True)[:10]:
                print(f"  {subj:<20} Total={info['total']:.4f}  Views={info['views']}")

        return {"primary": primary, "cross": cross, "detail": dict(detail)}

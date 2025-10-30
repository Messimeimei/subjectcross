# -*- coding: utf-8 -*-
"""
SubjectCalculator: 多维学科综合计算类（支持多策略）
-------------------------------------------------
策略：
 - default：强度 intensity + 广度 breadth + 平衡性 balance 综合打分
 - weighted：五维外部权重融合（各维内部归一化 + 融合排序）
 - 基于统计分布的动态交叉学科选择
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
            # 新增统计分布参数
            use_statistical_cross: bool = True,
            min_relative_threshold: float = 0.3,
            min_absolute_threshold: float = 0.05,
            max_gap_ratio: float = 0.5
    ):
        """
        初始化计算器
        - alpha, beta：调节广度和平衡性的影响
        - rel_thr, cover_thr, topk_cross：交叉筛选控制参数
        - strategy：计算策略（default / weighted）
        - use_statistical_cross：是否使用基于统计分布的交叉学科选择
        - min_relative_threshold：最小相对阈值（主学科得分的比例）
        - min_absolute_threshold：最小绝对阈值
        - max_gap_ratio：最大允许差距比例
        """
        self.alpha = alpha
        self.beta = beta
        self.rel_thr = rel_thr
        self.cover_thr = cover_thr
        self.topk_cross = topk_cross
        self.strategy = strategy
        self.debug = debug

        # 统计分布参数
        self.use_statistical_cross = use_statistical_cross
        self.min_relative_threshold = min_relative_threshold
        self.min_absolute_threshold = min_absolute_threshold
        self.max_gap_ratio = max_gap_ratio

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
    # ------------------ 统计分布辅助函数 ------------------------
    # ==========================================================

    def _calculate_score_statistics(self, scores):
        """计算得分分布的统计特征"""
        if len(scores) <= 1:
            return {
                'mean': scores[0] if scores else 0,
                'std': 0, 'q1': 0, 'q3': 0, 'iqr': 0,
                'cv': 0, 'gaps': [], 'max_gap': 0
            }

        scores_array = np.array(scores)

        # 基础统计量
        mean_score = np.mean(scores_array)
        std_score = np.std(scores_array)
        q1 = np.percentile(scores_array, 25)
        q3 = np.percentile(scores_array, 75)
        iqr = q3 - q1

        # 变异系数（离散程度的相对度量）
        cv = std_score / mean_score if mean_score > 0 else 0

        # 得分差距序列
        gaps = [scores[i] - scores[i + 1] for i in range(len(scores) - 1)]
        max_gap = max(gaps) if gaps else 0

        return {
            'mean': mean_score,
            'std': std_score,
            'q1': q1,
            'q3': q3,
            'iqr': iqr,
            'cv': cv,  # 变异系数，值越大说明分布越分散
            'gaps': gaps,
            'max_gap': max_gap
        }

    def _find_natural_breakpoint(self, scores, stats):
        """寻找得分分布的自然断点"""
        if len(scores) <= 2:
            return len(scores)  # 没有明显断点

        gaps = stats['gaps']
        if not gaps:
            return len(scores)

        # 计算平均差距
        mean_gap = np.mean(gaps)

        # 如果存在明显大于平均差距的断层
        for i, gap in enumerate(gaps):
            if gap > mean_gap * 2.0 and gap > stats['max_gap'] * 0.5:
                return i + 1  # 返回断层位置

        return len(scores)  # 没有明显断点

    def _dynamic_select_cross_subjects(self, ranked, primary_score):
        """基于统计分布动态选择交叉学科"""
        if len(ranked) <= 1:
            return []

        scores = [score for _, score in ranked]
        stats = self._calculate_score_statistics(scores)

        # 动态阈值
        relative_threshold = primary_score * self.min_relative_threshold
        absolute_threshold = self.min_absolute_threshold

        # 自然断点
        gap_threshold_idx = self._find_natural_breakpoint(scores, stats)
        # 关键改动 A：不要在断点=1时直接拦掉第2名
        # 只有当 i >= 2 时才应用断点限制，确保 i=1 的候选能被正常评估
        apply_break_after = max(2, gap_threshold_idx)

        cross_subjects = []
        # 关键改动 B：并列容差（绝对+相对取较大者，避免浮点抖动）
        tie_eps = max(1e-8, 1e-3 * primary_score)

        for i in range(1, min(len(ranked), 10)):  # 最多看前10名
            subject, score = ranked[i]

            # 绝对/相对阈值
            if score < absolute_threshold:
                continue
            if score < relative_threshold:
                continue

            # 应用断点（但不对 i < 2 生效，见上）
            if i >= apply_break_after:
                continue

            # 与前一名差距条件
            if i > 1:
                prev_score = ranked[i - 1][1]
                if prev_score > 0 and (prev_score - score) / prev_score > self.max_gap_ratio:
                    continue

            # 分布特征选择
            if stats['cv'] > 0.8:  # 分散
                if score > stats['mean'] + stats['std']:
                    cross_subjects.append(subject)
            elif stats['cv'] > 0.3:  # 中等分散
                if score >= stats['q1'] - 1e-12:
                    cross_subjects.append(subject)
            else:  # 集中
                if i <= 3:
                    cross_subjects.append(subject)

            # 并列补齐（关键改动 B）：把与当前入选项同分的后续并列项一并纳入
            if cross_subjects:
                base_score = score
                j = i + 1
                while j < len(ranked) and len(cross_subjects) < self.topk_cross:
                    sj, vj = ranked[j]
                    # 并列判定：|vj - base_score| <= tie_eps
                    if abs(vj - base_score) <= tie_eps and vj >= absolute_threshold and vj >= relative_threshold:
                        cross_subjects.append(sj)
                        j += 1
                    else:
                        break

            if len(cross_subjects) >= self.topk_cross:
                break

        # 保底策略：若仍为空，纳入“第二名及其并列项”
        if not cross_subjects and len(ranked) > 1:
            second_subject, second_score = ranked[1]
            if second_score >= primary_score * 0.6:  # 你的原有 60% 规则
                cross_subjects.append(second_subject)
                # 关键改动 C：把与第二名并列者也纳入（不超过 topk）
                k = 2
                while k < len(ranked) and len(cross_subjects) < self.topk_cross:
                    sj, vj = ranked[k]
                    if abs(vj - second_score) <= tie_eps and vj >= absolute_threshold and vj >= relative_threshold:
                        cross_subjects.append(sj)
                        k += 1
                    else:
                        break

        # 限制最大数量
        max_cross = min(self.topk_cross, len(ranked) - 1)
        return cross_subjects[:max_cross]

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
            - incites: softmax
            - refs: softmax
            - openalex: softmax
            - author: softmax
        ✅ 自动补全 detail 维度（缺省填 0）
        ✅ 基于统计分布的动态交叉学科选择
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
        primary, primary_score = ranked[0]

        # 动态选择交叉学科
        if self.use_statistical_cross:
            cross = self._dynamic_select_cross_subjects(ranked, primary_score)
        else:
            # 原有的固定选择方式
            cross = [d for d, _ in ranked[1:self.topk_cross + 1]]

        # ✅ 自动补全所有 subject 的 5 个维度（缺省填 0）
        all_dims = list(self.view_configs.keys())
        for subj in detail:
            for dim in all_dims:
                if dim not in detail[subj]["views"]:
                    detail[subj]["views"][dim] = 0.0

        # ✅ Debug 输出（可选）
        if self.debug:
            scores = [score for _, score in ranked]
            stats = self._calculate_score_statistics(scores)

            print(f"\n🧩 综合加权结果：主学科={primary}({primary_score:.4f})，交叉学科{len(cross)}个")
            print(f"   统计特征: 均值={stats['mean']:.4f}, 标准差={stats['std']:.4f}, 变异系数={stats['cv']:.4f}")
            print(
                f"   相对阈值: {primary_score * self.min_relative_threshold:.4f}, 绝对阈值: {self.min_absolute_threshold:.4f}")

            for subj, info in sorted(detail.items(), key=lambda kv: kv[1]["total"], reverse=True)[:10]:
                status = "主" if subj == primary else "交" if subj in cross else "未"
                print(f"  {status} {subj:<20} Total={info['total']:.4f}  Views={info['views']}")

        return {"primary": primary, "cross": cross, "detail": dict(detail)}
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
import joblib
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
            topk_cross: int = 5,
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

        # ---------- 新增：LR 模型加载 ----------
        self.lr_model = None
        self.lr_subjects = None  # 学科标签顺序
        if strategy == "lr":
            self._load_lr_model()


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
    # -------------------  加载 LR 模型  ------------------------
    # ==========================================================
    def _load_lr_model(self):
        """
        加载五维融合 LR 模型：
        - models/lr_model/lr_model.pkl           （sklearn LogisticRegression）
        - models/lr_model/global_stats.json      （各维度全局 min/max）
        - models/lr_model/best_params.json       （threshold / topk）
        """
        from pathlib import Path
        import joblib

        # 当前文件：utils/subject_calculator.py
        root = Path(__file__).resolve().parents[1]   # -> 项目根目录 subjectcross
        model_dir = root / "models" / "lr_model"

        model_pkl   = model_dir / "lr_model.pkl"
        stats_json  = model_dir / "global_stats.json"
        config_json = model_dir / "best_params.json"

        if not model_pkl.exists() or not stats_json.exists() or not config_json.exists():
            raise FileNotFoundError(
                "[LR ERROR] 缺少模型文件，请先运行训练脚本：python -m data.scripts.linear_regression"
            )

        # 1) 加载 LR 模型
        self.lr_model = joblib.load(model_pkl)

        # 2) 加载全局 min/max
        with open(stats_json, "r", encoding="utf-8") as f:
            stats = json.load(f)
        # stats 形如：{"incites": [min, max], "title_abs": [min, max], ...}
        self.lr_dim_min = {k: float(v[0]) for k, v in stats.items()}
        self.lr_dim_max = {k: float(v[1]) for k, v in stats.items()}

        # 3) 加载最优 threshold / topk（目前主要用来调参，可不用在推理中强制过滤）
        with open(config_json, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.lr_threshold = float(cfg.get("threshold", 0.5))
        self.lr_topk      = int(cfg.get("topk", self.topk_cross))

        print(f"[LR] 模型加载成功：{model_pkl}")
        print(f"[LR] global_stats 来自：{stats_json}")
        print(f"[LR] best_params: threshold={self.lr_threshold}, topk={self.lr_topk}")


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
        """
        寻找“宽松”的自然断点：
        - 用相邻差 gaps 的均值 + 0.5*std 作为阈值；
        - 返回断层后一项的索引（即 [0..idx-1] 为断层前，idx 为断层后第一项）；
        - 若未找到断点，返回 len(scores) 表示“无断点，保留到末尾”。
        说明：保持原签名与返回语义，兼容现有调用。
        """
        gaps = stats.get('gaps', [])
        if not gaps or len(scores) <= 2:
            return len(scores)

        import numpy as np
        gaps_arr = np.array(gaps, dtype=float)
        mean_gap = gaps_arr.mean()
        std_gap = gaps_arr.std()
        # 宽松阈值：均值 + 0.5*std（比原本 mean*2 更容易触发“保留更少”的断点）
        thr = mean_gap * 2 + 0.5 * std_gap

        for i, g in enumerate(gaps):
            if g >= thr:
                # 返回断层位置的“后一项”索引（与原实现一致：i 是 gap 的位置，对应保留到 i 之前，i+1 为后续起点）
                return i + 1

        # 没有明显断点：保留到末尾
        return len(scores)

    def _dynamic_select_cross_subjects(self, ranked, primary_score):
        """
        仅依据两条规则选交叉学科：
        1) 自然断点（宽松）：保留到第一处显著断层的“后一项”为止（至少保留到第2名）
        2) 相邻降幅约束：在上述保留段内，自前向后检查 (prev-curr)/prev，
           一旦超过 self.max_gap_ratio，则从该名开始及其后续全部舍弃（提前截断）
        备注：
          - 不再使用绝对/相对分阈值（min_absolute_threshold / min_relative_threshold）
          - 仍然尊重 self.topk_cross 上限
          - 仅返回标签列表（与原外部使用保持一致）
        """
        n = len(ranked)
        if n <= 1:
            return []

        # 构造分数序列（降序）
        scores = [s for _, s in ranked]

        # 计算统计量并寻找自然断点
        stats = self._calculate_score_statistics(scores)
        break_after = self._find_natural_breakpoint(scores, stats)  # 返回“断层后一项”的索引
        # 至少保留到第二名（索引1）；break_after 代表“可保留的右开界”，候选区间为 [1 .. break_after-1]
        keep_right = max(2, break_after)  # 右开界下限为2，确保第二名不会被断掉

        # 在 [1 .. keep_right-1] 区间内应用“相邻降幅约束”
        # 从第三名开始检查（i=2），一旦超过阈值就把区间截断到 i-1
        end_idx = keep_right - 1
        for i in range(2, keep_right):
            prev = scores[i - 1]
            curr = scores[i]
            if prev > 0 and (prev - curr) / prev > self.max_gap_ratio:
                end_idx = i - 1
                break

        # 交叉学科：从第二名（索引1）到 end_idx 的标签
        if end_idx < 1:
            return []
        cross_subjects = [subj for subj, _ in ranked[1: end_idx + 1]]

        # 限制最大数量
        if len(cross_subjects) > self.topk_cross:
            cross_subjects = cross_subjects[:self.topk_cross]

        return cross_subjects

    # ==========================================================
    # ------------------ 主调度入口 ----------------------------
    # ==========================================================
    def calc(self, row: dict):
        if self.strategy == "lr":
            return self._calc_lr(row)
        elif self.strategy == "weighted":
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
            "incites": "minmax",
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


    # ==========================================================
    # -------------------   策略 3：LR   ------------------------
    # ==========================================================
    def _calc_lr(self, row: dict):
        """
        LR 策略推理逻辑（与 linear_regression 训练脚本保持一致）：

        1. 从当前 row 里解析五个视角：
           - list_incites_direction
           - list_title_abs
           - list_author_aff_qwen
           - list_openalex
           - list_ref
           每个视角得到若干二元组 (原始学科名称, score)

        2. 抽取 4 位学科代码（例如 "1205 Library..." -> "1205"），
           对同一学科代码取最大得分，得到每个维度的：
               dim -> {学科代码: 原始得分}

        3. 使用训练时保存的全局 min/max 做 min-max 归一化：
               norm = (score - min) / (max - min)

        4. 对于当前 row 中所有出现过的学科代码 f，构造 5 维特征：
               [incites_norm, title_abs_norm, author_aff_norm, openalex_norm, refs_norm]
           喂给 self.lr_model.predict_proba 得到 prob(f 为“相关学科”的概率)

        5. 按 prob 降序排序：
           - 主学科 = 概率最高者
           - 交叉学科 = 复用 _dynamic_select_cross_subjects 的“动态断点”逻辑

        返回：
            {"primary": "1205", "cross": ["0710", ...], "detail": {...}}
        """
        if self.lr_model is None:
            raise RuntimeError("[LR ERROR] LR 模型未加载，请检查 __init__ 中 strategy='lr' 时是否正确调用 _load_lr_model()")

        # ---------- 1) 解析五个维度，聚合到“4位学科代码”层面 ----------
        dim_cols = {
            "incites":    "list_incites_direction",
            "title_abs":  "list_title_abs",
            "author_aff": "list_author_aff_qwen",
            "openalex":   "list_openalex",
            "refs":       "list_ref",
        }

        def extract_code(name: str) -> str:
            """从字符串中提取前4位数字作为学科代码，如 '1205 Library...' -> '1205'"""
            digits = "".join(ch for ch in str(name) if ch.isdigit())
            return digits[:4] if len(digits) >= 4 else ""

        from collections import defaultdict

        # dim_values: {dim_name: {subject_code: raw_score}}
        dim_values = {}
        for dim, col in dim_cols.items():
            raw = self._safe_eval_list(row.get(col, "[]"))
            flat = self._flatten_nested(raw)  # [(subj, score), ...] or 更深嵌套已被展开

            agg = defaultdict(float)
            for subj, val in flat:
                code = extract_code(subj)
                if not code:
                    continue
                try:
                    score = float(val)
                except Exception:
                    continue
                # 对同一学科代码取最大得分（也可以改为累加，看你训练时的习惯）
                if score > agg[code]:
                    agg[code] = score

            dim_values[dim] = agg

        # 收集该篇论文中所有出现过的学科代码
        subjects = set()
        for agg in dim_values.values():
            subjects.update(agg.keys())

        if not subjects:
            return {"primary": None, "cross": [], "detail": {}}

        # ---------- 2) 根据 global_stats 做 min-max 归一化 ----------
        def norm_value(dim: str, subj: str) -> float:
            raw_dict = dim_values.get(dim, {})
            if subj not in raw_dict:
                return 0.0
            val = raw_dict[subj]
            mn = self.lr_dim_min.get(dim, 0.0)
            mx = self.lr_dim_max.get(dim, 1.0)
            if mx <= mn:
                return 0.0
            return (val - mn) / (mx - mn)

        scores_dict = {}
        detail = {}

        # ---------- 3) 对每个学科代码跑 LR，得到概率 ----------
        for subj in subjects:
            feat = np.array([
                norm_value("incites",    subj),
                norm_value("title_abs",  subj),
                norm_value("author_aff", subj),
                norm_value("openalex",   subj),
                norm_value("refs",       subj),
            ], dtype=float)

            prob = float(self.lr_model.predict_proba(feat.reshape(1, -1))[0][1])
            scores_dict[subj] = prob
            detail[subj] = {
                "total": prob,
                "views": {
                    "incites":    float(feat[0]),
                    "title_abs":  float(feat[1]),
                    "author_aff": float(feat[2]),
                    "openalex":   float(feat[3]),
                    "refs":       float(feat[4]),
                }
            }

        if not scores_dict:
            return {"primary": None, "cross": [], "detail": {}}

        # ---------- 4) 排序 + 主/交叉学科选择 ----------
        ranked = sorted(scores_dict.items(), key=lambda kv: kv[1], reverse=True)
        primary, primary_score = ranked[0]

        # 交叉学科：采用与 weighted 策略相同的“动态断点”逻辑
        cross = self._dynamic_select_cross_subjects(ranked, primary_score)

        # （可选）Debug 输出
        if self.debug:
            scores_list = [s for _, s in ranked]
            stats = self._calculate_score_statistics(scores_list)
            print(f"\n[LR] 主学科 = {primary} ({primary_score:.4f})，交叉学科 = {cross}")
            print(f"     统计特征：mean={stats['mean']:.4f}, std={stats['std']:.4f}, cv={stats['cv']:.4f}")

        return {"primary": primary, "cross": cross, "detail": detail}

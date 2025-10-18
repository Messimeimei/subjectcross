# -*- coding: utf-8 -*-
"""
批量运行：从 CSV 输入，按 batch 计算每篇论文的 117 学科分数字典并输出 Top2 学科
依赖模块（你已经提供）：
  - lfs.vector2discipline.VectorDisciplineScorer
  - lfs.direction2discipline.Direction2Discipline
  - lfs.cite2discipline.CitationDisciplineScorer  (USE_CITATION=true 时启用)
"""

import os
import ast
import json
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import pandas as pd
from pathlib import Path
import os
import gc
import torch

from lfs.cite2discipline import CitationDisciplineScorer
from lfs.direction2discipline import Direction2Discipline
from lfs.vector2discipline import VectorDisciplineScorer, cache_path

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

BASE_DIR = Path(__file__).resolve().parents[1]  # /home/messi/pyprojects/subjectcross
os.chdir(BASE_DIR)  # 强制切换到项目根目录
print(f"📂 当前工作目录：{os.getcwd()}")

# ==== 读 env ====
from dotenv import load_dotenv
load_dotenv()

EMB_MODEL_NAME = os.getenv("EMB_MODEL_NAME", str(BASE_DIR / "models/bge-m3"))
DISC_CSV_PATH  = os.getenv("CSV_PATH", str(BASE_DIR / "data/zh_disciplines.csv"))
DISC_JSON_PATH = os.getenv("JSON_PATH", str(BASE_DIR / "data/zh_discipline_intro.json"))
CACHE_DIR      = os.getenv("CACHE_DIR", str(BASE_DIR / "models/bge-m3/.cache_embeddings"))
os.makedirs(CACHE_DIR, exist_ok=True)

# 向量化/检索类参数（vector2discipline 内部会读 env；此处只管 CSV 批处理大小）
DEFAULT_CSV_BATCH_SIZE = int(os.getenv("BATCH_SIZE", 64))

# 保留学科数
TOPN = int(os.getenv('TOPN', 3))

# 是否启用引用
USE_CITATION          = os.getenv("USE_CITATION").lower() == "true"
CITATION_MAX_WORKERS  = int(os.getenv("CITATION_MAX_WORKERS", 8))

# 5 路权重（启用引用时生效）
W_AFFIL     = float(os.getenv("W_AFFIL"))
W_JOURNAL   = float(os.getenv("W_JOURNAL"))
W_TITLEABS  = float(os.getenv("W_TITLEABS"))
W_DIRECTION = float(os.getenv("W_DIRECTION"))
W_CITATION  = float(os.getenv("W_CITATION"))

# 4 路权重（禁用引用时生效）
W4_AFFIL     = float(os.getenv("W4_AFFIL",     "0.05"))
W4_JOURNAL   = float(os.getenv("W4_JOURNAL",   "0.05"))
W4_TITLEABS  = float(os.getenv("W4_TITLEABS",  "0.5"))
W4_DIRECTION = float(os.getenv("W4_DIRECTION", "0.4"))

# 全局缓存 SentenceTransformer 模型（避免重复加载）
_GLOBAL_MODEL = None

def get_global_runner():
    global _GLOBAL_MODEL
    if _GLOBAL_MODEL is None:
        print("🧠 正在加载全局向量模型（仅加载一次）...")
        _GLOBAL_MODEL = BatchCSVRunner(use_gpu=True)
    return _GLOBAL_MODEL


# ========= 小工具 =========
def safe_str(row, key: str, default: str = "") -> str:
    v = row.get(key, default)
    try:
        if pd.isna(v):  # type: ignore
            return default
    except Exception:
        pass
    return str(v)

def _topn_mask(d: Dict[str, float], n: int) -> Dict[str, float]:
    if n is None or n <= 0:
        return d
    items = sorted(d.items(), key=lambda x: x[1], reverse=True)
    allow = {k for k, _ in items[:n]}
    return {k: (v if k in allow else 0.0) for k, v in d.items()}

def _sum_dicts(dicts: List[Dict[str, float]], weights: List[float]) -> Dict[str, float]:
    out = defaultdict(float)
    for d, w in zip(dicts, weights):
        if w <= 0 or not d:
            continue
        for k, v in d.items():
            out[k] += w * float(v)
    return dict(out)


# ========= 主流程 =========
class BatchCSVRunner:
    """
    负责：
      1) 加载/缓存学科嵌入库
      2) 批量对 3 路相似度来源做 GPU 向量化 + FAISS 批量检索
      3) 方向均分（批量）
      4) 引用聚合（可选，完全由 USE_CITATION 控制）
      5) 按权重融合（USE_CITATION 决定权重集合），输出每篇论文 Top2 学科 + 原始字段
    """
    def __init__(self, use_gpu: bool = True):
        # 相似度打分器 & 学科库
        self.vec = VectorDisciplineScorer(EMB_MODEL_NAME, use_gpu=use_gpu)
        self.code2name, self.code2intro = self.vec.load_disciplines(DISC_CSV_PATH, DISC_JSON_PATH)
        cpath = cache_path(EMB_MODEL_NAME, DISC_CSV_PATH, DISC_JSON_PATH)
        self.emb, self.codes, self.names, self.texts = self.vec.ensure_cache(cpath, self.code2name, self.code2intro)

        # 方向
        self.dir_scorer = Direction2Discipline(DISC_CSV_PATH)

        # 引用（是否启用由 USE_CITATION 决定；不用缓存判断）
        self.use_citation = USE_CITATION
        if self.use_citation:
            self.cite_scorer = CitationDisciplineScorer()
        else:
            self.cite_scorer = None

        # 117 学科全零骨架
        self.zero117 = {f"{c} {self.code2name[c]}": 0.0 for c in self.code2name.keys()}

    # ---- 3 路相似度：批量一次性计算 ----
    def _batch_three_sim_sources(
            self,
            batch_affils: List[List],
            batch_journals: List[str],
            batch_titles: List[str],
            batch_abstracts: List[str],
    ):
        def _fill_all(d: Dict[str, float]) -> Dict[str, float]:
            full = self.zero117.copy()
            full.update(d)
            return full

        # 机构文本拼接
        affil_texts = []
        for row_affils in batch_affils:
            acc = []
            for a in (row_affils or []):
                if isinstance(a, (list, tuple)) and len(a) == 2:
                    acc.append(str(a[1]))
                elif isinstance(a, str):
                    acc.append(a)
            affil_texts.append(" ".join(acc) if acc else "")

        journal_texts = [j or "" for j in batch_journals]
        ta_texts = [f"{(t or '')}\n{(a or '')}" for t, a in zip(batch_titles, batch_abstracts)]

        s_affil = [_fill_all(d) for d in
                   self.vec.score_batch(affil_texts, self.codes, self.names, self.emb)]
        torch.cuda.empty_cache()
        gc.collect()

        s_journal = [_fill_all(d) for d in
                     self.vec.score_batch(journal_texts, self.codes, self.names, self.emb)]
        torch.cuda.empty_cache()
        gc.collect()

        s_ta = [_fill_all(d) for d in
                self.vec.score_batch(ta_texts, self.codes, self.names, self.emb)]
        torch.cuda.empty_cache()
        gc.collect()

        return s_affil, s_journal, s_ta

    def _batch_direction_scores(self, batch_directions: List[str]):
        outs = []

        def _fill_all(d: Dict[str, float]) -> Dict[str, float]:
            full = self.zero117.copy()
            full.update(d)
            return full

        for s in batch_directions:
            d = self.dir_scorer.direction_to_scores(s or "")
            outs.append(_fill_all(d))
        return outs

    # ---- 引用分数（批量；完全由 USE_CITATION 决定是否参与） ----
    def _batch_citation_scores(self, batch_reference_dois: List[List[str]]) -> List[Dict[str, float]]:
        if not self.use_citation:
            return [self.zero117.copy() for _ in batch_reference_dois]

        outs: List[Dict[str, float]] = [self.zero117.copy() for _ in batch_reference_dois]
        with ThreadPoolExecutor(max_workers=CITATION_MAX_WORKERS) as ex:
            futs = {}
            for i, dois in enumerate(batch_reference_dois):
                futs[ex.submit(self.cite_scorer.process_doi_list, dois or [])] = i
            for f in as_completed(futs):
                i = futs[f]
                try:
                    d = f.result()
                except Exception as e:
                    print(f"⚠️ 引用打分失败（第{i}条），已置零：{e}")
                    d = self.zero117.copy()
                outs[i] = d
        return outs

    # ---- 对一个 batch 融合生成结果（平滑融合版，与 Integrated 一致）----
    def fuse_batch(
        self,
        batch_affils: List[List],
        batch_journals: List[str],
        batch_titles: List[str],
        batch_abstracts: List[str],
        batch_directions: List[str],
        batch_reference_dois: Optional[List[List[str]]] = None
    ) -> List[Dict[str, float]]:
        n = len(batch_titles)
        assert all(len(x) == n for x in [batch_affils, batch_journals, batch_abstracts, batch_directions]), "各输入列表长度需一致"

        # 分别计算各来源分数
        s_affil, s_journal, s_ta = self._batch_three_sim_sources(
            batch_affils, batch_journals, batch_titles, batch_abstracts
        )
        s_dir = self._batch_direction_scores(batch_directions)
        if batch_reference_dois is None:
            batch_reference_dois = [[] for _ in range(n)]
        s_cit = self._batch_citation_scores(batch_reference_dois)
        print(f"机构， 期刊，标题摘要， 研究方向")
        print(len(s_affil), len(s_journal), len(s_ta), len(s_dir))
        print(s_affil[0], s_journal[0], s_ta[0], s_dir[0])

        fused_list: List[Dict[str, float]] = []
        for i in range(n):
            # 初始化全量 117 学科分数
            fused_scores = self.zero117.copy()

            if self.use_citation:
                dicts   = [s_affil[i], s_journal[i], s_ta[i], s_dir[i], s_cit[i]]
                weights = [W_AFFIL,    W_JOURNAL,    W_TITLEABS,   W_DIRECTION, W_CITATION]
            else:
                dicts   = [s_affil[i], s_journal[i], s_ta[i], s_dir[i]]
                weights = [W4_AFFIL,   W4_JOURNAL,   W4_TITLEABS,  W4_DIRECTION]

            # 正规化权重
            total_w = sum(weights)
            for d, w in zip(dicts, weights):
                if not d or w <= 0:
                    continue
                w_norm = w / total_w
                for k in fused_scores:
                    fused_scores[k] += d.get(k, 0.0) * w_norm

            fused_list.append(fused_scores)
        return fused_list


def process_csv_in_batches(
    csv_file: str,
    batch_size: int = DEFAULT_CSV_BATCH_SIZE
) -> List[Dict]:
    """
    读取 CSV，按 batch 计算每篇论文的 117 学科分字典，返回 list[dict]
    每个 dict 包含：
      DOI, 来源, 研究方向, 论文标题, CR_学科, CR_摘要, CR_作者机构, CR_参考文献DOI, top2_disciplines
    """
    df = pd.read_csv(csv_file)
    runner = get_global_runner()
    results: List[Dict] = []

    for start in range(0, len(df), batch_size):
        sub = df.iloc[start:start+batch_size]

        # ——收集批输入——
        batch_affils, batch_journals, batch_titles = [], [], []
        batch_abstracts, batch_dirs, batch_cites = [], [], []

        for _, row in sub.iterrows():
            journal   = safe_str(row, "来源")
            title     = safe_str(row, "论文标题")
            abstract  = safe_str(row, "CR_摘要")
            direction = safe_str(row, "研究方向")
            aff_raw   = safe_str(row, "CR_作者和机构")
            dois_raw  = safe_str(row, "CR_参考文献DOI")

            try:
                affils = ast.literal_eval(aff_raw) if aff_raw else []
            except Exception:
                affils = []
            try:
                dois = ast.literal_eval(dois_raw) if dois_raw else []
            except Exception:
                dois = []

            batch_journals.append(journal)
            batch_titles.append(title)
            batch_abstracts.append(abstract)
            batch_dirs.append(direction)
            batch_affils.append(affils)
            batch_cites.append(dois)

        # ——批量融合——
        fused_list = runner.fuse_batch(
            batch_affils=batch_affils,
            batch_journals=batch_journals,
            batch_titles=batch_titles,
            batch_abstracts=batch_abstracts,
            batch_directions=batch_dirs,
            batch_reference_dois=batch_cites
        )

        # ——组装输出——
        for i, (_, row) in enumerate(sub.iterrows()):
            fused = fused_list[i]
            top3 = sorted([(k, v) for k, v in fused.items() if v > 0], key=lambda x: -x[1])[:TOPN]
            topn_disciplines = [{"discipline": k, "score": float(v)} for k, v in top3]

            results.append({
                "DOI": safe_str(row, "DOI"),
                "来源": safe_str(row, "来源"),
                "研究方向": safe_str(row, "研究方向"),
                "论文标题": safe_str(row, "论文标题"),
                "CR_摘要": safe_str(row, "CR_摘要"),
                "CR_作者和机构": safe_str(row, "CR_作者和机构"),
                "CR_参考文献DOI": safe_str(row, "CR_参考文献DOI"),
                "topn_disciplines": topn_disciplines
            })

        print(f"✅ 已处理 {min(start+batch_size, len(df))}/{len(df)}")

    return results

def save_results_to_csv(results: List[Dict], out_path=''):
    """
    把批量运行的结果保存到 CSV 文件
    - results: process_csv_in_batches 的输出 list[dict]
    - out_path: 输出路径，例如 "../output/results.csv"
    """
    # 转换成 DataFrame（topn_disciplines 需要转成 JSON 字符串保存）
    rows = []
    for r in results:
        row = r.copy()
        row["topn_disciplines"] = json.dumps(r["topn_disciplines"], ensure_ascii=False)
        rows.append(row)
    df_out = pd.DataFrame(rows)

    # 保存到 CSV
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"💾 已保存结果到 {out_path}")




# ========= CLI =========
if __name__ == "__main__":
    csv_name = '0825 Aerospace Science and Technology.csv'
    out = process_csv_in_batches(f"data/02crossref_data/{csv_name}", batch_size=DEFAULT_CSV_BATCH_SIZE)
    save_results_to_csv(out, f"data/03subject_data/{csv_name}")
    # 打印前几条示例
    for r in out[:10]:
        print(json.dumps(r, ensure_ascii=False, indent=2))

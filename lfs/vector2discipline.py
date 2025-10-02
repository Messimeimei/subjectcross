# -*- coding: utf-8 -*-
"""
VectorDisciplineScorer
- 对117个一级学科的“学科介绍chunk”向量化（缓存）
- 提供单条与批量的文本→学科相似度计算
- FAISS 支持批量检索
- 重要：始终返回 117 学科全量分数（不做任何 TopN 掩码）
"""

import os, json, re
from typing import List, Tuple, Dict
from collections import defaultdict
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ========= 环境变量 =========
EMB_MODEL_NAME = os.getenv("EMB_MODEL_NAME", "../models/bge-m3")
CSV_PATH = os.getenv("CSV_PATH", "../data/zh_disciplines_with_code.csv")
JSON_PATH = os.getenv("JSON_PATH", "../data/zh_discipline_intro_with_code.json")
CACHE_DIR = os.getenv("CACHE_DIR", "../models/bge-m3/.cache_embeddings")
os.makedirs(CACHE_DIR, exist_ok=True)

BATCH_SIZE = int(os.getenv("BATCH_SIZE", 128))
USE_FP16 = os.getenv("USE_FP16", "true").lower() == "true"
TARGET_CH_LEN = int(os.getenv("TARGET_CH_LEN", 220))
TOPK_EACH_DISC = int(os.getenv("TOPK_EACH_DISC", 3))   # 聚合某学科时取该学科命中的前K个chunk的平均
MAX_RETRIEVE = int(os.getenv("MAX_RETRIEVE", 2000))    # FAISS 检索返回的候选上限


# ========= 工具函数 =========
def get_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

def _hash_of(text: str) -> str:
    import hashlib
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def cache_path(model_name: str, csv_path: str, json_path: str) -> str:
    sig = json.dumps({
        "model": model_name,
        "csv": os.path.abspath(csv_path),
        "json": os.path.abspath(json_path),
        "params": {"TARGET_CH_LEN": TARGET_CH_LEN}
    }, ensure_ascii=False)
    return os.path.join(CACHE_DIR, f"chunks_{_hash_of(sig)}.npz")

def split_chinese_intro(text: str, target_len: int = TARGET_CH_LEN):
    if not text or not isinstance(text, str):
        return []
    s = re.sub(r"\r\n?", "\n", text).strip()
    seps = "。！？；;\n"
    buf, pieces = "", []
    for ch in s:
        buf += ch
        if ch in seps and len(buf) >= target_len:
            pieces.append(buf.strip()); buf = ""
    if buf.strip():
        pieces.append(buf.strip())
    merged, cur = [], ""
    for p in pieces:
        if len(cur) + len(p) < target_len // 2:
            cur += p
        else:
            if cur: merged.append(cur); cur = ""
            merged.append(p)
    if cur: merged.append(cur)
    merged = [m for m in merged if len(m) >= 20]
    if not merged and s:
        merged = [s[:max(200, target_len)]]
    return merged


# ========= 主类 =========
class VectorDisciplineScorer:
    def __init__(self, model_path=EMB_MODEL_NAME, use_gpu=True):
        device = get_device()
        self.device = "cuda" if use_gpu and device == "cuda" else "cpu"
        self.model = SentenceTransformer(model_path, device=self.device)
        if self.device == "cuda" and USE_FP16:
            try:
                self.model = self.model.half()
            except Exception:
                pass

    # ---------- 学科库 ----------
    def load_disciplines(self, csv_path=CSV_PATH, json_path=JSON_PATH):
        df = pd.read_csv(csv_path, sep=r"\s+", header=None, names=["code", "name"], engine="python")
        df["code"] = df["code"].astype(str).str.zfill(4)
        code2name = dict(df.values)
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        code2intro = {code: v.get("intro", "") for code, v in raw.items()}
        return code2name, code2intro

    def build_chunks(self, code2name, code2intro):
        rows = []
        for code, name in code2name.items():
            intro = code2intro.get(code, "")
            chunks = split_chinese_intro(intro, TARGET_CH_LEN)
            for c in chunks:
                rows.append((code, name, c))
        return rows

    def embed_texts(self, texts: List[str], batch_size=BATCH_SIZE) -> np.ndarray:
        vecs = self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=False,
            normalize_embeddings=True, convert_to_numpy=True
        )
        return np.asarray(vecs, dtype="float32")

    def ensure_cache(self, cache_file: str, code2name, code2intro, force_rebuild=False):
        if os.path.exists(cache_file) and not force_rebuild:
            data = np.load(cache_file, allow_pickle=True)
            return data["emb"], data["codes"].tolist(), data["names"].tolist(), data["texts"].tolist()
        chunks = self.build_chunks(code2name, code2intro)
        texts = [t[2] for t in chunks]
        emb = self.embed_texts(texts)
        codes = [t[0] for t in chunks]
        names = [t[1] for t in chunks]
        np.savez_compressed(cache_file, emb=emb,
                            codes=np.array(codes, dtype=object),
                            names=np.array(names, dtype=object),
                            texts=np.array(texts, dtype=object))
        return emb, codes, names, texts

    # ---------- FAISS 检索（支持批量 query） ----------
    def _faiss_index(self, dim):
        import faiss
        try:
            if faiss.get_num_gpus() > 0 and self.device == "cuda":
                res = faiss.StandardGpuResources()
                return faiss.GpuIndexFlatIP(res, dim)
            else:
                return faiss.IndexFlatIP(dim)
        except Exception:
            return faiss.IndexFlatIP(dim)

    def faiss_search_batch(self, query_vecs: np.ndarray, emb: np.ndarray, topk: int):
        """
        批量检索：query_vecs shape=(N, dim)
        返回：D shape=(N, topk), I shape=(N, topk)
        """
        import faiss
        dim = emb.shape[1]
        index = self._faiss_index(dim)
        index.add(emb)
        if query_vecs.ndim == 1:
            query_vecs = query_vecs.reshape(1, -1)
        topk = min(topk, emb.shape[0])
        D, I = index.search(query_vecs, topk)
        return D, I

    # ---------- 打分核心（始终返回 117 学科全量） ----------
    def _aggregate_scores_by_code(
        self,
        I_row: np.ndarray,
        D_row: np.ndarray,
        codes: List[str],
        names: List[str],
    ) -> Dict[str, float]:
        """
        把一条 query 的检索结果聚合到“学科代码 学科名”维度。
        - 先把属于同一学科代码的多个 chunk 相似度做 TopK 均值（TOPK_EACH_DISC）。
        - 返回 117 学科全量分数（未命中=0.0，不做任何 TopN 掩码）。
        """
        # 余弦（内积）[-1,1] → [0,1]
        dense_norm = (D_row + 1.0) / 2.0
        by_disc = defaultdict(list)
        for pos, idx in enumerate(I_row):
            by_disc[codes[idx]].append(float(dense_norm[pos]))

        code2name = {c: n for c, n in zip(codes, names)}
        scores_full: Dict[str, float] = {}
        for code in codes:
            vals = by_disc.get(code, [])
            if vals:
                vals_sorted = sorted(vals, reverse=True)[:max(1, TOPK_EACH_DISC)]
                score = float(np.mean(vals_sorted))
            else:
                score = 0.0
            scores_full[f"{code} {code2name[code]}"] = score
        return scores_full

    # ---------- 单条接口 ----------
    def _score_single_text(self, text: str, codes, names, emb) -> Dict[str, float]:
        q = self.embed_texts([text])[0]
        D, I = self.faiss_search_batch(q, emb, min(MAX_RETRIEVE, emb.shape[0]))
        return self._aggregate_scores_by_code(I[0], D[0], codes, names)

    def score_from_affiliations(self, authors_affils: List[Tuple[str, str]], codes, names, texts, emb) -> Dict[str, float]:
        affils = []
        for a in authors_affils:
            if isinstance(a, (list, tuple)) and len(a) == 2:
                affils.append(a[1])
            elif isinstance(a, str):
                affils.append(a)
        text = " ".join(affils) if affils else ""
        return self._score_single_text(text, codes, names, emb)

    def score_from_journal(self, journal_name: str, codes, names, texts, emb) -> Dict[str, float]:
        return self._score_single_text(journal_name or "", codes, names, emb)

    def score_from_title_abstract(self, title: str, abstract: str, codes, names, texts, emb) -> Dict[str, float]:
        return self._score_single_text((title or "") + "\n" + (abstract or ""), codes, names, emb)

    # ---------- 批量接口 ----------
    def score_batch(self, texts: List[str], codes, names, emb) -> List[Dict[str, float]]:
        """
        批量输入文本 → 返回每条文本的 117 学科全量分数字典
        """
        if not texts:
            return []
        q_vecs = self.embed_texts(texts)  # GPU 批量向量化
        D, I = self.faiss_search_batch(q_vecs, emb, min(MAX_RETRIEVE, emb.shape[0]))  # FAISS 批量检索
        out = []
        for r in range(q_vecs.shape[0]):
            out.append(self._aggregate_scores_by_code(I[r], D[r], codes, names))
        return out


# ========= 使用示例（可选）=========
if __name__ == "__main__":
    scorer = VectorDisciplineScorer(EMB_MODEL_NAME, use_gpu=True)
    code2name, code2intro = scorer.load_disciplines(CSV_PATH, JSON_PATH)
    cpath = cache_path(EMB_MODEL_NAME, CSV_PATH, JSON_PATH)
    emb, codes, names, texts = scorer.ensure_cache(cpath, code2name, code2intro)

    # 单条（返回 117 全量）
    r = scorer.score_from_journal("Science Education", codes, names, texts, emb)
    print("=== 单条结果 ===")
    for k, v in r.items():
        print(f"{k}: {v:.4f}")
    print("🔥 Top3:", sorted(r.items(), key=lambda x: x[1], reverse=True)[:3])

    # 批量（返回 117 全量）
    batch_texts = [
        "Deep learning for protein structure prediction",
        "Machine learning approaches in clinical epidemiology",
    ]
    res = scorer.score_batch(batch_texts, codes, names, emb)

    print("\n=== 批量结果 ===")
    for i, d in enumerate(res, 1):
        print(f"\n样本 {i} 全部学科分数：")
        for k, v in d.items():
            print(f"{k}: {v:.4f}")
        print(f"🔥 Top3: {sorted(d.items(), key=lambda x: x[1], reverse=True)[:3]}")

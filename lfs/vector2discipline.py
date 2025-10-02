# paper_discipline_scorer.py
# -*- coding: utf-8 -*-
"""
统一的学科打分器
支持三种输入：
- 作者机构
- 期刊名称
- 标题+摘要
同时支持单篇和批量模式
"""

import os, json, re, hashlib
from typing import List, Tuple, Dict
from collections import defaultdict
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ========= 环境变量配置 =========
EMB_MODEL_NAME = os.getenv("EMB_MODEL_NAME", "../models/bge-m3")
CSV_PATH = os.getenv("CSV_PATH", "../data/zh_disciplines_with_code.csv")
JSON_PATH = os.getenv("JSON_PATH", "../data/zh_discipline_intro_with_code.json")
CACHE_DIR = os.getenv("CACHE_DIR", "../models/bge-m3/.cache_embeddings")
os.makedirs(CACHE_DIR, exist_ok=True)

BATCH_SIZE = int(os.getenv("BATCH_SIZE", 128))
USE_FP16 = os.getenv("USE_FP16", "true").lower() == "true"
TARGET_CH_LEN = int(os.getenv("TARGET_CH_LEN", 220))
TOPK_EACH_DISC = int(os.getenv("TOPK_EACH_DISC", 3))
MAX_RETRIEVE = int(os.getenv("MAX_RETRIEVE", 2000))


# ========= 工具函数 =========
def get_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _hash_of(text: str) -> str:
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
            pieces.append(buf.strip())
            buf = ""
    if buf.strip():
        pieces.append(buf.strip())

    merged, cur = [], ""
    for p in pieces:
        if len(cur) + len(p) < target_len // 2:
            cur += p
        else:
            if cur:
                merged.append(cur)
                cur = ""
            merged.append(p)
    if cur:
        merged.append(cur)

    merged = [m for m in merged if len(m) >= 20]
    if not merged and s:
        merged = [s[:max(200, target_len)]]
    return merged


# ========= 主类 =========
class PaperDisciplineScorer:
    def __init__(self, model_path=EMB_MODEL_NAME, use_gpu=True):
        # 根据参数决定是否用 GPU
        device = get_device()
        self.device = "cuda" if use_gpu and device == "cuda" else "cpu"
        self.model = SentenceTransformer(model_path, device=self.device)
        if self.device == "cuda" and USE_FP16:
            try:
                self.model = self.model.half()
            except Exception:
                pass

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
        """批量向量化文本"""
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,   # 关闭 tqdm
            normalize_embeddings=True,
            convert_to_numpy=True
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

    def faiss_search(self, query_vec: np.ndarray, emb: np.ndarray, topk: int):
        import faiss
        dim = emb.shape[1]
        try:
            if faiss.get_num_gpus() > 0 and self.device == "cuda":
                res = faiss.StandardGpuResources()
                index = faiss.GpuIndexFlatIP(res, dim)
            else:
                index = faiss.IndexFlatIP(dim)
        except Exception:
            index = faiss.IndexFlatIP(dim)
        index.add(emb)
        D, I = index.search(query_vec.reshape(1, -1), topk)
        return D[0], I[0]

    def topk_mean(self, scores, k=TOPK_EACH_DISC):
        if not scores:
            return 0.0
        s = sorted(scores, reverse=True)[:max(1, k)]
        return float(np.mean(s))

    # ========= 单输入接口 =========
    def score_from_affiliations(self, authors_affils: List[Tuple[str, str]], codes, names, texts, emb, topn=3):
        affil_texts = []
        for item in authors_affils:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                affil_texts.append(item[1])
            elif isinstance(item, str):
                affil_texts.append(item)
        text = " ".join(affil_texts) if affil_texts else ""
        return self._score(text, codes, names, emb, topn)

    def score_from_journal(self, journal_name: str, codes, names, texts, emb, topn=3):
        return self._score(journal_name, codes, names, emb, topn)

    def score_from_title_abstract(self, title: str, abstract: str, codes, names, texts, emb, topn=3):
        return self._score((title or "") + "\n" + (abstract or ""), codes, names, emb, topn)

    def _score(self, q_text: str, codes, names, emb, topn):
        """单条文本打分"""
        q_vec = self.embed_texts([q_text])[0]
        return self._score_core(q_vec, codes, names, emb, topn)

    # ========= 批量接口 =========
    def score_batch(self, texts: List[str], codes, names, emb, topn=3):
        """批量文本打分"""
        if not texts:
            return []
        q_vecs = self.embed_texts(texts)   # (N, dim)
        results = []
        for q_vec in q_vecs:
            results.append(self._score_core(q_vec, codes, names, emb, topn))
        return results

    def _score_core(self, q_vec, codes, names, emb, topn):
        """核心相似度计算（供单条/批量共用）"""
        D, I = self.faiss_search(q_vec, emb, min(MAX_RETRIEVE, emb.shape[0]))
        dense_norm = (D + 1.0) / 2.0
        by_disc = defaultdict(list)
        for pos, idx in enumerate(I):
            by_disc[codes[idx]].append(float(dense_norm[pos]))
        code2name = {c: n for c, n in zip(codes, names)}
        scores = {}
        for code in codes:
            vals = by_disc.get(code, [])
            score = self.topk_mean(vals, TOPK_EACH_DISC) if vals else 0.0
            scores[f"{code} {code2name[code]}"] = score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        topn_keys = {k for k, _ in sorted_scores[:topn]}
        return {k: (v if k in topn_keys else 0.0) for k, v in scores.items()}


# ========= 使用示例 =========
if __name__ == "__main__":
    scorer = PaperDisciplineScorer(EMB_MODEL_NAME, use_gpu=True)
    code2name, code2intro = scorer.load_disciplines(CSV_PATH, JSON_PATH)
    cpath = cache_path(EMB_MODEL_NAME, CSV_PATH, JSON_PATH)
    emb, codes, names, texts = scorer.ensure_cache(cpath, code2name, code2intro)

    # 单条示例
    print("\n=== 单条示例 ===")
    print(scorer.score_from_journal("Science Education", codes, names, texts, emb))

    # 批量示例
    print("\n=== 批量示例 ===")
    batch_texts = [
        "Deep learning for protein structure prediction",
        "Machine learning approaches in clinical epidemiology",
    ]
    res = scorer.score_batch(batch_texts, codes, names, emb, topn=3)
    for i, r in enumerate(res, 1):
        print(f"样本 {i}:", list(r.items())[:5])

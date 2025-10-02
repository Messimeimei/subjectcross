# -*- coding: utf-8 -*-
"""
CitationDisciplineScorer
功能：
1. 输入 DOI 列表
2. 调用 Crossref API 获取元数据
3. 调用 PaperDisciplineScorer 计算学科分数（机构 / 期刊 / 标题摘要）
4. 汇总参考文献的学科分布（平均分）
5. 最终输出：117 学科全量字典（key="代码 中文名"），仅保留 TopK，其余置 0
"""

import os, re, json, requests
from typing import List, Dict, Tuple
from collections import defaultdict
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

from lfs.vector2discipline import PaperDisciplineScorer, cache_path

# ========= 环境变量 =========
load_dotenv()
EMB_MODEL_NAME = os.getenv("EMB_MODEL_NAME", "../models/bge-m3")
CSV_PATH = os.getenv("CSV_PATH", "../data/zh_disciplines_with_code.csv")
JSON_PATH = os.getenv("JSON_PATH", "../data/zh_discipline_intro_with_code.json")


def strip_tags(text: str) -> str:
    """去掉 XML/JATS 标签"""
    if not text:
        return ""
    txt = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", txt).strip()


def get_crossref_metadata(doi: str) -> Dict:
    """根据 DOI 获取 Crossref 精简元数据"""
    base_url = "https://api.crossref.org/works/"
    url = f"{base_url}{doi}"
    try:
        headers = {"User-Agent": "Mozilla/5.0 (mailto:your_email@example.com)"}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        msg = response.json().get("message", {})
        return {
            "DOI": doi,
            "title": (msg.get("title", [""])[0] if msg.get("title") else "").strip(),
            "abstract": strip_tags(msg.get("abstract", "")),
            "journal": msg.get("container-title", [""])[0] if msg.get("container-title") else "",
            "authors_affils": [
                (f"{a.get('given','')} {a.get('family','')}".strip(),
                 (a.get("affiliation", [{}])[0].get("name", "") if a.get("affiliation") else ""))
                for a in msg.get("author", []) or []
            ]
        }
    except Exception as e:
        print(f"⚠️ 获取 DOI 元数据失败: {doi}, 错误: {e}")
        return {"DOI": doi, "title": "", "abstract": "", "journal": "", "authors_affils": []}


class CitationDisciplineScorer:
    def __init__(self, weights: Dict[str, float] = None):
        """
        :param weights: 三个来源的权重 {"affil": , "journal": , "titleabs": }
                        默认 (0.1, 0.3, 0.6)
        """
        self.weights = weights or {"affil": 0.1, "journal": 0.3, "titleabs": 0.6}

        # 初始化打分类
        self.scorer = PaperDisciplineScorer(EMB_MODEL_NAME)
        self.code2name, self.code2intro = self.scorer.load_disciplines(CSV_PATH, JSON_PATH)
        cpath = cache_path(EMB_MODEL_NAME, CSV_PATH, JSON_PATH)
        self.emb, self.codes, self.names, self.texts = self.scorer.ensure_cache(
            cpath, self.code2name, self.code2intro, force_rebuild=False
        )
        # 反向映射（中文名 -> 代码），以备不时之需
        self.name2code: Dict[str, str] = {v: k for k, v in self.code2name.items()}
        self._code_pattern = re.compile(r"^\s*(\d{4})")

    # ---------- 关键：统一 key 为代码 ----------
    def _key_to_code(self, k: str) -> str:
        """
        将各种形式的 key（如 '0712', '0712 科学技术史', '科学技术史'）统一转为学科代码。
        匹配不到则返回空字符串。
        """
        if not k:
            return ""
        k = str(k).strip()
        # 1) 直接是代码
        if k in self.code2name:
            return k
        # 2) 形如 '0712 科学技术史' / '0712 xxx'
        m = self._code_pattern.match(k)
        if m:
            code = m.group(1)
            return code if code in self.code2name else ""
        # 3) 纯中文名
        return self.name2code.get(k, "")

    def _normalize_score_dict(self, d: Dict[str, float]) -> Dict[str, float]:
        """
        将任意 key（代码/代码+中文/中文）统一为 {代码: 分数}
        如果同一代码出现多次，取最大值（也可改为求和/平均，按需）。
        """
        out: Dict[str, float] = {}
        for k, v in (d or {}).items():
            code = self._key_to_code(k)
            if not code:
                continue
            if code in out:
                out[code] = max(out[code], float(v))
            else:
                out[code] = float(v)
        return out

    # ------------------------------------------

    def process_doi_list(self, dois: List[str], topn_each_paper: int = 3, keep_topk: int = 3) -> Dict[str, float]:
        """
        根据 DOI 列表 → 获取元数据 → 学科打分
        最终返回：长度=117 的 {"代码 中文名": 分数} 字典（仅保留全局平均分TopK，其余置0）
        """
        # 1. 并行获取 Crossref 元数据
        print(f"🚀 正在并行获取 Crossref 元数据，共 {len(dois)} 篇参考文献...")
        metas = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(get_crossref_metadata, doi) for doi in dois]
            for f in as_completed(futures):
                metas.append(f.result())
        print(f"✅ 已完成 Crossref 获取，成功 {len(metas)} 篇")

        # 2. 每篇计算加权分数（先把各来源分数字典规范成“代码”）
        discipline_scores: Dict[str, List[float]] = defaultdict(list)

        for meta in metas:
            s_affil_raw = self.scorer.score_from_affiliations(
                meta["authors_affils"], self.codes, self.names, self.texts, self.emb, topn=topn_each_paper
            )
            s_journal_raw = self.scorer.score_from_journal(
                meta["journal"], self.codes, self.names, self.texts, self.emb, topn=topn_each_paper
            )
            s_ta_raw = self.scorer.score_from_title_abstract(
                meta["title"], meta["abstract"], self.codes, self.names, self.texts, self.emb, topn=topn_each_paper
            )

            # 统一成 “代码: 分数”
            s_affil = self._normalize_score_dict(s_affil_raw)
            s_journal = self._normalize_score_dict(s_journal_raw)
            s_ta = self._normalize_score_dict(s_ta_raw)

            # 合并加权（按代码）
            union_codes = set(s_affil) | set(s_journal) | set(s_ta)
            combined_by_code: Dict[str, float] = {}
            for code in union_codes:
                combined_by_code[code] = (
                    self.weights["affil"]   * s_affil.get(code, 0.0) +
                    self.weights["journal"] * s_journal.get(code, 0.0) +
                    self.weights["titleabs"]* s_ta.get(code, 0.0)
                )

            # 取该篇 TopN（按代码）
            if combined_by_code:
                top_disciplines = sorted(combined_by_code.items(), key=lambda x: x[1], reverse=True)[:topn_each_paper]
                for code, score in top_disciplines:
                    # 收集到全局池
                    if code in self.code2name:  # 只收录 117 学科之内的代码
                        discipline_scores[code].append(float(score))

        # 3. 计算全局平均分（代码 → 平均分）
        avg_scores: Dict[str, float] = {code: (sum(v) / len(v)) for code, v in discipline_scores.items() if v}

        # 4. 构造 117 学科全量输出字典（key="代码 中文名"）
        final_scores_117: Dict[str, float] = {f"{c} {self.code2name[c]}": 0.0 for c in self.code2name.keys()}
        if avg_scores:
            topk_global: List[Tuple[str, float]] = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)[:keep_topk]
            # 仅保留 TopK，其余为 0
            for code, score in topk_global:
                key = f"{code} {self.code2name.get(code, '')}"
                if key in final_scores_117:
                    final_scores_117[key] = float(score)
        else:
            print("⚠️ 未得到任何平均分（可能 Crossref 元数据过少/无命中）。")

        return final_scores_117


# ========= 使用示例 =========
if __name__ == "__main__":
    scorer = CitationDisciplineScorer()
    dois = ['10.1177/0963662506064240',
            '10.1080/09500693.2012.664293',
            '10.1002/sce.20078',
            '10.1002/tea.21227']
    final_scores = scorer.process_doi_list(dois, topn_each_paper=3, keep_topk=3)
    print(json.dumps(final_scores, indent=2, ensure_ascii=False))

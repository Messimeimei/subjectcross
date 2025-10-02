# -*- coding: utf-8 -*-
"""
综合学科打分器
支持五种输入：
- 作者机构
- 期刊名称
- 标题+摘要
- 研究方向字符串
- DOI 列表
输出：
1) fused_scores → {学科代码 学科名: 综合分数}，117 学科全量
2) top3 → 前3学科及其分数（带学科名）
"""
import ast
import os, json
from typing import Dict
from dotenv import load_dotenv

from lfs.cite2discipline import CitationDisciplineScorer
from lfs.direction2discipline import Direction2Discipline
from lfs.vector2discipline import PaperDisciplineScorer, cache_path

load_dotenv()

# ========= 环境变量 =========
EMB_MODEL_NAME = os.getenv("EMB_MODEL_NAME", "../models/bge-m3")
CSV_PATH = os.getenv("CSV_PATH", "../data/zh_disciplines_with_code.csv")
JSON_PATH = os.getenv("JSON_PATH", "../data/zh_discipline_intro_with_code.json")
EN_DISC_FILE = os.getenv("EN_DISC_FILE", "../data/zh_disciplines_with_code.csv")


class IntegratedDisciplineScorer:
    def __init__(self, weights: Dict[str, float] = None):
        """
        :param weights: 五个来源的权重 {"affil": , "journal": , "titleabs": , "direction": , "doi": }
        """
        self.weights = weights or {
            "affil": 0.05,
            "journal": 0.1,
            "titleabs": 0.5,
            "direction": 0.3,
            "doi": 0.05
        }

        # 初始化打分类
        self.paper_scorer = PaperDisciplineScorer(EMB_MODEL_NAME)
        self.direction_scorer = Direction2Discipline(EN_DISC_FILE)
        self.doi_scorer = CitationDisciplineScorer()

        # 加载学科数据（中文 117个）
        self.code2name, self.code2intro = self.paper_scorer.load_disciplines(CSV_PATH, JSON_PATH)
        cpath = cache_path(EMB_MODEL_NAME, CSV_PATH, JSON_PATH)
        self.emb, self.codes, self.names, self.texts = self.paper_scorer.ensure_cache(
            cpath, self.code2name, self.code2intro, force_rebuild=False
        )

    def fuse_scores(self, affils=None, journal=None, title=None, abstract=None,
                    direction=None, dois=None, topn=3) -> Dict[str, float]:
        """
        输入多种信息，输出加权融合后的 117 学科分数
        """
        # 初始化全量 117 学科分数
        fused_scores = {f"{c} {n}": 0.0 for c, n in self.code2name.items()}
        total_w = sum(self.weights.values())

        # 作者机构
        if affils:
            s = self.paper_scorer.score_from_affiliations(
                affils, self.codes, self.names, self.texts, self.emb, topn=topn
            )
            w = self.weights["affil"] / total_w
            for c, n in self.code2name.items():
                fused_scores[f"{c} {n}"] += s.get(c, 0.0) * w

        # 期刊
        if journal:
            s = self.paper_scorer.score_from_journal(
                journal, self.codes, self.names, self.texts, self.emb, topn=topn
            )
            w = self.weights["journal"] / total_w
            for c, n in self.code2name.items():
                fused_scores[f"{c} {n}"] += s.get(c, 0.0) * w

        # 标题+摘要
        if title or abstract:
            s = self.paper_scorer.score_from_title_abstract(
                title, abstract, self.codes, self.names, self.texts, self.emb, topn=topn
            )
            w = self.weights["titleabs"] / total_w
            for c, n in self.code2name.items():
                fused_scores[f"{c} {n}"] += s.get(c, 0.0) * w

        # 研究方向
        if direction:
            s_dir = self.direction_scorer.direction_to_scores(direction)
            # s_dir 已经是 {代码 学科名: 分数} 格式（117学科），直接融合
            w = self.weights["direction"] / total_w
            for k in fused_scores:
                fused_scores[k] += s_dir.get(k, 0.0) * w

        # DOI 列表
        if dois:
            s_doi = self.doi_scorer.process_doi_list(dois, topn_each_paper=topn, keep_topk=3)
            w = self.weights["doi"] / total_w
            for k in fused_scores:
                fused_scores[k] += s_doi.get(k, 0.0) * w

        return fused_scores

    def get_topn(self, scores: Dict[str, float], n: int = 3):
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]


# ========= 使用示例 =========
if __name__ == "__main__":
    scorer = IntegratedDisciplineScorer()

    affils = [('Nick Riggle', 'Department of Philosophy, University of San Diego, San Diego, CA, USA')]
    journal = "JOURNAL OF AESTHETICS AND ART CRITICISM"
    title = "Toward a Communitarian Theory of Aesthetic Value"
    abstract = "AbstractOur paradigms of aesthetic value condition the philosophical questions we pose and hope to answer about it. Theories of aesthetic value are typically individualistic, in the sense that the paradigms they are designed to capture, and the questions to which they are offered as answers, center the individual’s engagement with aesthetic value. Here I offer some considerations that suggest that such individualism is a mistake and sketch a communitarian way of posing and answering questions about the nature of aesthetic value."
    direction = "1304 Fine Art; 0101 Philosophy; 1301 Art Theory"
    dois = ['10.2307/2026797', '10.1111/phpr.12641', '10.1353/hph.2014.0020', '10.1111/phpr.12727', '10.1093/acprof:oso/9780199691517.003.0011', '10.1093/oso/9780197625798.001.0001', '10.1093/aesthj/ayaa006', '10.1080/24740500.2017.1287034', '10.1086/662744', '10.1093/aesthj/ayz024', '10.1080/00048400903207104', '10.1093/aesthj/ayy044', '10.1111/phpr.12747', '10.1111/j.1468-0114.2012.01429.x', '10.1111/j.1540-6245.2012.01519.x', '10.1111/j.1747-9991.2006.00007.x', '10.1093/aesthj/ays038']

    fused_scores = scorer.fuse_scores(
        affils=affils,
        journal=journal,
        title=title,
        abstract=abstract,
        direction=direction,
        dois=dois,
        topn=3
    )

    print("✅ 综合分数 Top5:")
    for k, v in list(fused_scores.items())[:5]:
        print(k, v)

    print("\n🔥 Top3 学科:")
    for k, v in scorer.get_topn(fused_scores, n=2):
        print(k, v)

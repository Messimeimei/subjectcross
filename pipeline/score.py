# -*- coding: utf-8 -*-
"""
综合学科打分器, 支持输入一条论文数据进行计算
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
from lfs.vector2discipline import VectorDisciplineScorer, cache_path

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
        self.paper_scorer = VectorDisciplineScorer(EMB_MODEL_NAME)
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
                affils, self.codes, self.names, self.texts, self.emb
            )
            w = self.weights["affil"] / total_w
            for c, n in self.code2name.items():
                fused_scores[f"{c} {n}"] += s.get(c, 0.0) * w

        # 期刊
        if journal:
            s = self.paper_scorer.score_from_journal(
                journal, self.codes, self.names, self.texts, self.emb,
            )
            w = self.weights["journal"] / total_w
            for c, n in self.code2name.items():
                fused_scores[f"{c} {n}"] += s.get(c, 0.0) * w

        # 标题+摘要
        if title or abstract:
            s = self.paper_scorer.score_from_title_abstract(
                title, abstract, self.codes, self.names, self.texts, self.emb
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

    affils = [('Nathalie Percie du Sert', 'Head of Experimental Design and Reporting, NC3Rs, London, United Kingdom'), ('Viki Hurst', 'Science Manager – Experimental Design and Reporting, NC3Rs, London, United Kingdom'), ('Amrita Ahluwalia', 'Professor of Vascular Pharmacology, Co-Director, The William Harvey Research Institute, London, United Kingdom'), ('Amrita Ahluwalia', 'Director of the Barts Cardiovascular CTU, Queen Mary University of London, London, United Kingdom'), ('Sabina Alam', 'Director of Publishing Ethics and Integrity, Taylor & Francis Group, London, United Kingdom'), ('Marc T. Avey', 'Lead Health Scientist, Health Science Practice, ICF, Durham, North Carolina, United States of America'), ('Monya Baker', 'Senior Editor, Opinion, Nature, San Francisco, California, United States of America'), ('William J. Browne', 'Professor of Statistics, School of Education, University of Bristol, Bristol, United Kingdom'), ('Alejandra Clark', 'Senior Editor, Team Manager – Life Sciences, PLOS ONE, Cambridge, United Kingdom'), ('Innes C. Cuthill', 'Professor of Behavioural Ecology, School of Biological Sciences, University of Bristol, Bristol, United Kingdom'), ('Ulrich Dirnagl', 'Director, QUEST Center for Transforming Biomedical Research, Berlin Institute of Health & Department of Experimental Neurology, Charite Universitätsmedizin Berlin, Berlin, Germany'), ('Michael Emerson', 'Reader in Platelet Pharmacology, National Heart and Lung Institute, Imperial College London, London, United Kingdom'), ('Paul Garner', 'Professor, and Director of the Centre for Evidence Synthesis in Global Health, Clinical Sciences Department, Liverpool School of Tropical Medicine, Liverpool, United Kingdom'), ('Stephen T. Holgate', 'MRC Clinical Professor, Clinical and Experimental Sciences, University of Southampton, Southampton, United Kingdom'), ('David W. Howells', 'Professor of Neuroscience and Brain Plasticity, Tasmanian School of Medicine, University of Tasmania, Hobart, Australia'), ('Natasha A. Karp', 'Principal Scientist – Statistician & UK Team Lead, Data Sciences & Quantitative Biology, Discovery Sciences, R&D, AstraZeneca, Cambridge, United Kingdom'), ('Stanley E. Lazic', 'Chief Scientific Officer, Prioris.ai Inc, Ottawa, Canada'), ('Katie Lidster', 'Programme Manager – Animal Welfare, NC3Rs, London, United Kingdom'), ('Catriona J. MacCallum', 'Director of Open Science, Hindawi Ltd, London, United Kingdom'), ('Malcolm Macleod', 'Professor of Neurology and Translational Neuroscience, Centre for Clinical Brain Sciences, University of Edinburgh, Edinburgh, United Kingdom'), ('Malcolm Macleod', 'Academic Lead for Research Improvement and Research Integrity, University of Edinburgh, Edinburgh, United Kingdom'), ('Esther J. Pearl', 'Programme Manager – Experimental Design, NC3Rs, London, United Kingdom'), ('Ole H. Petersen', 'Director of the Academia Europaea Knowledge Hub, Cardiff University, Cardiff, United Kingdom'), ('Frances Rawle', 'Director of Policy, Ethics and Governance, Medical Research Council, London, United Kingdom'), ('Penny Reynolds', 'Biostatistician, Statistics in Anesthesiology Research (STAR) Core & Research Assistant Professor, Department of Anesthesiology College of Medicine, University of Florida, Gainesville, Florida, United States of America'), ('Kieron Rooney', 'Associate Professor, Discipline of Exercise and Sport Science, Faculty of Medicine and Health, University of Sydney, Sydney, Australia'), ('Emily S. Sena', 'Stroke Association Kirby Laing Foundation Senior Non-Clinical Lecturer, Centre for Clinical Brain Sciences, University of Edinburgh, Edinburgh, United Kingdom'), ('Shai D. Silberberg', 'Director of Research Quality, National Institute of Neurological Disorders and Stroke, Bethesda, Maryland, United States of America'), ('Thomas Steckler', 'Associate Director, BRQC Animal Welfare Strategy Lead, Janssen Pharmaceutica NV, Beerse, Belgium'), ('Hanno Würbel', 'Professor for Animal Welfare, Veterinary Public Health Institute, Vetsuisse Faculty, University of Bern, Bern, Switzerland')]
    journal = "JOURNAL OF CEREBRAL BLOOD FLOW AND METABOLISM"
    title = "The ARRIVE guidelines 2.0: Updated guidelines for reporting animal research*"
    abstract = """
    Reproducible science requires transparent reporting. The ARRIVE guidelines (Animal Research: Reporting of In Vivo Experiments) were 
    originally developed in 2010 to improve the reporting of animal research. They consist of a checklist of information to include in 
    publications describing in vivo experiments to enable others to scrutinise the work adequately, evaluate its methodological rigour, 
    and reproduce the methods and results. Despite considerable levels of endorsement by funders and journals over the years, adherence to the guidelines has been inconsistent, and the anticipated improvements in the quality of reporting in animal research publications have not been achieved. Here, we introduce ARRIVE 2.0. The guidelines have been updated and information reorganised to facilitate their use in practice. We used a Delphi exercise to prioritise and divide the items of the guidelines into 2 sets, the “ARRIVE Essential 10,” which constitutes the minimum requirement, and the “Recommended Set,” which describes the research context. This division facilitates improved reporting of animal research by supporting a stepwise approach to implementation. This helps journal editors and reviewers verify that the most important items are being reported in manuscripts. We have also developed the accompanying Explanation and Elaboration document, which serves (1) to explain the rationale behind each item in the guidelines, (2) to clarify key concepts, and (3) to provide illustrative examples. We aim, through these changes, to help ensure that researchers, reviewers,
     and journal editors are better equipped to improve the rigour and transparency of the scientific process and thus reproducibility.
    """
    direction = "1001 Basic Medicine; 1002 Clinical Medicine"
    dois = ['10.1126/scitranslmed.aaf5027', '10.1161/CIRCRESAHA.114.303819', '10.12688/f1000research.11334.1', '10.1371/journal.pbio.1002456', '10.1038/d41586-018-06178-7', '10.1073/pnas.1708274114', '10.1111/ejn.13519', '10.1017/CBO9781139344319', '10.1371/journal.pbio.2003779', '10.1017/9781139696647', '10.1371/journal.pbio.1002273', '10.1161/STROKEAHA.108.525386', '10.1016/j.pain.2008.08.017', '10.1111/j.1751-0813.1995.tb07534.x', '10.1001/jama.296.14.1731', '10.1371/journal.pone.0007824', '10.1371/journal.pmed.1000245', '10.1016/S0140-6736(13)62228-X', '10.1038/483531a', '10.1080/17482960701856300', '10.1258/la.2010.0010021', '10.1002/jgm.1473', '10.1371/journal.pbio.1000412', '10.1113/jphysiol.2010.192278', '10.1113/expphysiol.2010.053793', '10.1111/j.1476-5381.2010.00873.x', '10.1111/j.1939-165X.2012.00418.x', '10.1016/j.joca.2012.02.010', '10.4103/0976-500X.72351', '10.1371/journal.pone.0166733', '10.1371/journal.pone.0197882', '10.1186/s41073-019-0069-3', '10.1136/bmjos-2017-000035', '10.1371/journal.pone.0183591', '10.1161/CIRCRESAHA.117.310628', '10.1371/journal.pone.0165999', '10.1371/journal.pone.0200303', '10.7120/09627286.28.1.107', '10.1371/journal.pbio.3000411', '10.1038/nature11556', '10.1038/nbt.2261', '10.1038/sdata.2016.18', '10.1371/journal.pmed.1000217', '10.1080/15265160903318343', '10.7717/peerj.3208', '10.1371/journal.pone.0175583', '10.1124/mol.119.118927', '10.1136/jme.2008.024299', '10.1007/s11948-007-9011-z', '10.1038/nmeth.2471', '10.1111/bph.14153', '10.1038/d41586-018-07245-9']
    fused_scores = scorer.fuse_scores(
        affils=affils,
        journal=journal,
        title=title,
        abstract=abstract,
        direction=direction,
        dois=dois,
        topn=3
    )

    print("\n🔥 Top3 学科:")
    for k, v in scorer.get_topn(fused_scores, n=3):
        print(k, v)

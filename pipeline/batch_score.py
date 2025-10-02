# -*- coding: utf-8 -*-
import ast
import os, json
import pandas as pd
from dotenv import load_dotenv
from pipeline.score import IntegratedDisciplineScorer

load_dotenv()

import pandas as pd

def safe_str(row, col_name: str, default: str = "") -> str:
    """
    从 DataFrame 的行中安全获取某一列，确保返回字符串。
    - 如果值为 NaN / None，则返回 default
    - 否则转成字符串
    """
    val = row.get(col_name, default)
    if pd.isna(val):
        return default
    return str(val)


def process_csv(csv_file: str, topn: int = 3, keep_topk: int = 3):
    """
    从 CSV 文件读取数据，逐行调用 IntegratedDisciplineScorer
    返回：每篇论文的 117 学科分数（dict），以及 TopK 学科
    """
    scorer = IntegratedDisciplineScorer()
    df = pd.read_csv(csv_file)

    results = []

    for _, row in df.iterrows():
        # 取字段
        doi = safe_str(row, "DOI")
        journal = safe_str(row, "来源")
        title = safe_str(row, "论文标题")
        abstract = safe_str(row, "CR_摘要")
        direction = safe_str(row, "研究方向")

        affils = row.get("CR_作者机构", "")
        affils = ast.literal_eval(affils)
        dois = row.get("CR_参考文献DOI", "")
        dois = ast.literal_eval(dois)     # 处理进来是字符串列表的情况

        # 🚀 调用综合打分器
        fused_scores = scorer.fuse_scores(
            affils=affils,
            journal=journal,
            title=title,
            abstract=abstract,
            direction=direction,
            dois=dois,
            topn=topn
        )

        topk = scorer.get_topn(fused_scores, n=keep_topk)

        results.append({
            "doi": doi,
            "journal": journal,
            "title": title,
            "topk": topk,
            "scores": fused_scores   # 117 学科全量
        })

    return results


# ========= 使用示例 =========
if __name__ == "__main__":
    csv_file = "../data/processed_data/1001 Basic Medicine_merged.csv"   # 你的输入 CSV 文件
    results = process_csv(csv_file, topn=2, keep_topk=2)

    # 打印前两篇的结果
    for r in results[:]:
        print("\n📄 DOI:", r["doi"])
        print("🔥 Top3:", r["topk"])

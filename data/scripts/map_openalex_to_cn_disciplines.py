# -*- coding: utf-8 -*-
"""
OpenAlex 学科 → 中国117一级学科 映射脚本（双语 + 缓存检测 + 自动合并）
--------------------------------------------------------
功能：
1. 同时读取中英文 OpenAlex 学科列表（无表头）
2. 分别计算与中国117学科的相似度映射（自动缓存判断）
3. 若已存在英文/中文映射 JSON 文件，直接跳过计算
4. 最后自动合并两份结果（英文键，值取并集）
--------------------------------------------------------
输入：
    data/fields_all_en.csv
    data/fields_all_cn.csv
输出：
    data/openalex_to_cn_disciplines_en.json
    data/openalex_to_cn_disciplines_cn.json
    data/openalex_to_cn_disciplines_merged.json
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# ---------- 自动添加项目根目录 ----------
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from lfs.vector2discipline import VectorDisciplineScorer, cache_path

# ---------- 加载 .env ----------
load_dotenv(ROOT / ".env")

# ===================== 配置区 =====================
FIELDS_EN = ROOT / "data/fields_all_en.csv"
FIELDS_CN = ROOT / "data/fields_all_cn.csv"

OUTPUT_EN = ROOT / "data/openalex_to_cn_disciplines_en.json"
OUTPUT_CN = ROOT / "data/openalex_to_cn_disciplines_cn.json"
OUTPUT_MERGED = ROOT / "data/openalex_to_cn_disciplines_merged.json"

TOPN = int(os.getenv("TOPN", 5))

CSV_PATH = ROOT / os.getenv("CSV_PATH", "data/zh_disciplines.csv")
JSON_PATH = ROOT / os.getenv("JSON_PATH", "data/zh_discipline_intro.json")
EMB_MODEL_NAME = os.getenv("EMB_MODEL_NAME", "models/bge-m3")
CACHE_DIR = ROOT / os.getenv("CACHE_DIR", "models/bge-m3/.cache_embeddings")


# ===================== 工具函数 =====================
def read_field_list(file_path: Path):
    """读取学科列表，无表头兼容纯文本"""
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    try:
        df = pd.read_csv(file_path)
        if "field_name" in df.columns:
            return df["field_name"].dropna().astype(str).tolist()
        else:
            return df.iloc[:, 0].dropna().astype(str).tolist()
    except Exception:
        with open(file_path, "r", encoding="utf-8") as f:
            return [line.strip().strip('"') for line in f if line.strip()]


def compute_mapping(field_list, scorer, codes, names, emb, topn=TOPN):
    """批量计算映射"""
    batch_scores = scorer.score_batch(field_list, codes, names, emb)
    result = {}
    for field, score_dict in zip(field_list, batch_scores):
        sorted_disc = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)[:topn]
        result[field] = [code_name for code_name, _ in sorted_disc]
    return result


def merge_results(en_data, cn_data):
    """按行顺序合并两份结果（键保留英文，值取并集）"""
    merged = {}
    en_keys = list(en_data.keys())
    cn_vals_list = list(cn_data.values())

    for i, key in enumerate(en_keys):
        en_vals = en_data.get(key, [])
        cn_vals = cn_vals_list[i] if i < len(cn_vals_list) else []
        merged_vals = list(dict.fromkeys(en_vals + cn_vals))
        merged[key] = merged_vals
    return merged


# ===================== 主函数 =====================
def main():
    print("📦 检查已有映射文件 ...")
    en_exists = OUTPUT_EN.exists()
    cn_exists = OUTPUT_CN.exists()

    # ---------- 若两者都存在，仅合并 ----------
    if en_exists and cn_exists:
        print("✅ 检测到中英文结果已存在，直接执行合并。")
        with open(OUTPUT_EN, "r", encoding="utf-8") as f:
            en_result = json.load(f)
        with open(OUTPUT_CN, "r", encoding="utf-8") as f:
            cn_result = json.load(f)
        merged = merge_results(en_result, cn_result)
        with open(OUTPUT_MERGED, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        print(f"🎉 合并完成，共 {len(merged)} 条，结果已保存至：{OUTPUT_MERGED}")
        return

    # ---------- 加载模型与117学科 ----------
    print("🚀 正在加载 VectorDisciplineScorer ...")
    scorer = VectorDisciplineScorer()
    print("📘 加载117学科信息 ...")
    code2name, code2intro = scorer.load_disciplines(str(CSV_PATH), str(JSON_PATH))
    print("💾 构建或加载嵌入缓存 ...")
    cpath = cache_path(EMB_MODEL_NAME, str(CSV_PATH), str(JSON_PATH))
    emb, codes, names, texts = scorer.ensure_cache(cpath, code2name, code2intro)

    # ---------- 英文 ----------
    if not en_exists:
        en_fields = read_field_list(FIELDS_EN)
        print(f"🧠 正在计算英文字段映射，共 {len(en_fields)} 条 ...")
        en_result = compute_mapping(en_fields, scorer, codes, names, emb)
        OUTPUT_EN.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_EN, "w", encoding="utf-8") as f:
            json.dump(en_result, f, ensure_ascii=False, indent=2)
        print(f"✅ 英文映射完成：{OUTPUT_EN}")
    else:
        with open(OUTPUT_EN, "r", encoding="utf-8") as f:
            en_result = json.load(f)
        print("⚙️ 已检测到英文结果文件，跳过计算。")

    # ---------- 中文 ----------
    if not cn_exists:
        cn_fields = read_field_list(FIELDS_CN)
        print(f"🧠 正在计算中文字段映射，共 {len(cn_fields)} 条 ...")
        cn_result = compute_mapping(cn_fields, scorer, codes, names, emb)
        with open(OUTPUT_CN, "w", encoding="utf-8") as f:
            json.dump(cn_result, f, ensure_ascii=False, indent=2)
        print(f"✅ 中文映射完成：{OUTPUT_CN}")
    else:
        with open(OUTPUT_CN, "r", encoding="utf-8") as f:
            cn_result = json.load(f)
        print("⚙️ 已检测到中文结果文件，跳过计算。")

    # ---------- 合并 ----------
    print("🔗 正在合并中英文结果 ...")
    merged = merge_results(en_result, cn_result)
    with open(OUTPUT_MERGED, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"🎉 合并完成，共 {len(merged)} 条，结果已保存至：{OUTPUT_MERGED}")


# ===================== 主入口 =====================
if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
OpenAlex Field/Subfield → 中国117一级学科映射类（支持 Embedding / LLM 双模式）
方案A：LLM批量模式，一次调用返回整张JSON映射表
-------------------------------------------------------------------
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from utils.llm_call import call_qwen_rank
from utils.vector2discipline import VectorDisciplineScorer, cache_path

# ====== 基础路径配置 ======
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

load_dotenv(ROOT / ".env")

# ====== 默认路径 ======
CSV_PATH = ROOT / os.getenv("CSV_PATH", "data/zh_disciplines.csv")
JSON_PATH = ROOT / os.getenv("JSON_PATH", "data/zh_discipline_intro.json")
EMB_MODEL_NAME = os.getenv("EMB_MODEL_NAME", "models/bge-m3")

FIELDS_EN = ROOT / "data/openalex_fields_en.csv"
SUBFIELDS_EN = ROOT / "data/openalex_subfields_en.csv"
MAP_DIR = ROOT / "data"
MAP_DIR.mkdir(parents=True, exist_ok=True)


# ========================== 工具函数 ==========================
def read_list(file_path: Path):
    """读取一列 csv 或 txt 文件"""
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    try:
        df = pd.read_csv(file_path, header=None)
        return df.iloc[:, 0].dropna().astype(str).tolist()
    except Exception:
        with open(file_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]


def save_json(data, path: Path):
    """保存 JSON 文件"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"💾 已保存: {path} ({len(data)} 条)")


# ========================== 主类 ==========================
class OpenAlexMapper:
    def __init__(self, csv_path=CSV_PATH, json_path=JSON_PATH, emb_model=EMB_MODEL_NAME):
        self.csv_path = csv_path
        self.json_path = json_path
        self.emb_model = emb_model
        self.scorer = None
        self.emb_cache = None
        print("✅ 初始化 OpenAlexMapper 完成。")

    # ---------- Embedding 方法 ----------
    def map_with_embedding(self, field_list, topn=5):
        """基于向量相似度映射"""
        if self.scorer is None:
            print("🧠 加载向量模型与学科缓存 ...")
            self.scorer = VectorDisciplineScorer()
            code2name, code2intro = self.scorer.load_disciplines(str(self.csv_path), str(self.json_path))
            cpath = cache_path(self.emb_model, str(self.csv_path), str(self.json_path))
            emb, codes, names, texts = self.scorer.ensure_cache(cpath, code2name, code2intro)
            self.emb_cache = (emb, codes, names, texts)
        emb, codes, names, texts = self.emb_cache

        print(f"⚙️ 开始计算 Embedding 相似度 (Top{topn}) ...")
        batch_scores = self.scorer.score_batch(field_list, codes, names, emb)
        result = {}
        for field, score_dict in zip(field_list, batch_scores):
            sorted_disc = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)[:topn]
            result[field] = [(f"{k}", float(v)) for k, v in sorted_disc]
        return result

    # ---------- LLM 批量方法 ----------
    def map_with_llm_batch(self, fields_en, subfields_en, all_disciplines, disc_intro_json, top_field=5, top_subfield=2):
        """
        使用 Qwen (discipline_map) 模型，一次性批量输入全部 OpenAlex fields+subfields，
        输出完整 JSON 映射表。
        """
        print(f"🧠 调用 Qwen discipline_map 模式：Fields={len(fields_en)}, Subfields={len(subfields_en)}")

        mapped = call_qwen_rank(
            text_block="",                     # 模板中不需要 text_block
            disciplines_json=all_disciplines,  # 中国117学科列表
            disciplines_intro_json=disc_intro_json,
            fields_en_list=fields_en,          # 填充 {{FIELDS_EN_LIST}}
            subfields_en_list=subfields_en,    # 填充 {{SUBFIELDS_EN_LIST}}
            topn_field=top_field,              # 填充 {{TOPN_FIELD}}
            topn_subfield=top_subfield,        # 填充 {{TOPN_SUBFIELD}}
            mode="discipline_map",
            max_retries=3
        )

        if isinstance(mapped, dict):
            print(f"✅ Qwen 批量映射完成，共 {len(mapped)} 条记录。")
            return mapped
        else:
            print("⚠️ Qwen 未返回 JSON 对象，结果为空。")
            return {}

    # ---------- 统一运行接口 ----------
    def run(self, mode="embedding", top_field=5, top_subfield=2):
        """运行 Embedding 或 LLM 模式"""
        print(f"🚀 模式：{mode.upper()} 开始运行 ...")

        all_disciplines = read_list(self.csv_path)
        field_en = read_list(FIELDS_EN)
        subfield_en = read_list(SUBFIELDS_EN)

        if mode == "embedding":
            map_field = self.map_with_embedding(field_en, top_field)
            map_subfield = self.map_with_embedding(subfield_en, top_subfield)

            prefix = f"{mode}_openalex_to_cn"
            save_json(map_field, MAP_DIR / f"{prefix}_field_en.json")
            save_json(map_subfield, MAP_DIR / f"{prefix}_subfield_en.json")

            merged = {**map_field, **map_subfield}
            save_json(merged, MAP_DIR / f"{prefix}_merged.json")
            print(f"🎉 Embedding 模式完成，共 {len(merged)} 条记录。")

        elif mode == "llm":
            # 大模型不使用学科介绍
            disc_intro = {}

            merged = self.map_with_llm_batch(field_en, subfield_en, all_disciplines, disc_intro, top_field, top_subfield)
            save_json(merged, MAP_DIR / "llm_openalex_to_cn_merged.json")
            print(f"🎯 LLM 批量模式完成，共 {len(merged)} 条记录。")

        else:
            raise ValueError("mode 必须是 'embedding' 或 'llm'")


# ========================== 主程序入口 ==========================
if __name__ == "__main__":
    mapper = OpenAlexMapper()

    # 运行词向量模式
    # mapper.run(mode='embedding', top_field=5, top_subfield=2)

    # 运行 LLM 批量模式
    mapper.run(mode="llm", top_field=5, top_subfield=2)

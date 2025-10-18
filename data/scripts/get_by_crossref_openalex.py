# -*- coding: utf-8 -*-
# Created by Messimeimei
# Updated by ChatGPT (2025/10)
"""
并发抓取 Crossref 元数据 + OpenAlex 学科主题 + 作者机构 + 参考文献 + 中国学科映射
"""

import random
import time
import os
import re
import json
import requests
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Dict, Any


class CrossrefMetaProcessor:
    def __init__(self, input_dir: str,
                 output_dir: str = "data/02crossref_data",
                 openalex_email: str = "3170529323@qq.com"):
        # ✅ 自动识别项目根路径（即 main.py 所在位置）
        ROOT_DIR = Path(__file__).resolve().parents[2]

        self.input_dir = str((ROOT_DIR / input_dir).resolve())
        self.output_dir = str((ROOT_DIR / output_dir).resolve())

        # ✅ 固定映射文件路径
        self.map_path = ROOT_DIR / "data/openalex_to_cn_disciplines_merged.json"

        self.openalex_email = openalex_email
        self.df_raw = None
        self.df_merged = None

    # ================================
    # 工具函数
    # ================================
    @staticmethod
    def _strip_tags(text: str) -> str:
        """去掉 JATS/XML 标签"""
        if not text:
            return ""
        txt = re.sub(r"<[^>]+>", "", text)
        txt = re.sub(r"\s+", " ", txt)
        return txt.strip()

    # ================================
    # Step 1. 读取所有 CSV
    # ================================
    def load_all_csvs(self) -> pd.DataFrame:
        """读取目录下所有 csv 并合并，清洗 DOI"""
        all_dfs = []
        for file in os.listdir(self.input_dir):
            if file.endswith(".csv"):
                path = os.path.join(self.input_dir, file)
                try:
                    df = pd.read_csv(path, encoding="utf-8-sig")
                    all_dfs.append(df)
                except Exception as e:
                    print(f"⚠️ 文件 {file} 读取失败: {e}")

        if not all_dfs:
            raise RuntimeError("❌ 输入目录下没有可用的 CSV 文件")

        df = pd.concat(all_dfs, ignore_index=True)
        cols = [c.strip().lower() for c in df.columns]
        doi_col = [df.columns[i] for i, c in enumerate(cols) if c == "doi"][0]
        source_col = [df.columns[i] for i, c in enumerate(cols) if "来源" in c][0]
        field_col = [df.columns[i] for i, c in enumerate(cols) if "研究方向" in c][0]
        title_col = [df.columns[i] for i, c in enumerate(cols) if "论文标题" in c][0]

        df[doi_col] = df[doi_col].astype(str).str.strip()
        df[doi_col] = df[doi_col].replace(["", "nan", "NaN", "None", "NULL"], pd.NA)
        df = df.dropna(subset=[doi_col])
        df = df[df[doi_col].str.contains("/", na=False)]
        df = df.drop_duplicates(subset=[doi_col]).reset_index(drop=True)
        self.df_raw = df[[doi_col, source_col, field_col, title_col]]
        return self.df_raw

    # ================================
    # Step 2. Crossref 抓取
    # ================================
    @staticmethod
    def get_crossref_metadata(doi: str) -> dict:
        """根据 DOI 获取 Crossref 元数据（摘要、作者、机构、参考文献）"""
        base_url = "https://api.crossref.org/works/"
        url = f"{base_url}{doi}"
        try:
            headers = {"User-Agent": "Mozilla/5.0 (mailto:3170529323@qq.com)"}
            response = requests.get(url, headers=headers, timeout=25)
            response.raise_for_status()
            msg = response.json().get("message", {})

            abstract_txt = CrossrefMetaProcessor._strip_tags(msg.get("abstract", ""))

            # ---- 作者及机构 ----
            author_info = []
            for a in msg.get("author", []) or []:
                name = (a.get("given", "") + " " + a.get("family", "")).strip()
                affs = [aff.get("name") for aff in a.get("affiliation", []) if aff.get("name")]
                if name or affs:
                    author_info.append({"name": name, "affiliation": affs})

            # ---- 参考文献 DOI ----
            ref_dois = []
            for ref in msg.get("reference", []) or []:
                doi_ref = ref.get("DOI") or ref.get("doi")
                if doi_ref:
                    ref_dois.append(doi_ref.strip())

            return {
                "DOI": doi,
                "CR_摘要": abstract_txt,
                "CR_出版商": msg.get("publisher", ""),
                "CR_作者和机构": json.dumps(author_info, ensure_ascii=False),
                "CR_参考文献DOI": json.dumps(ref_dois, ensure_ascii=False)
            }

        except Exception:
            return {
                "DOI": doi,
                "CR_摘要": "",
                "CR_出版商": "",
                "CR_作者和机构": "[]",
                "CR_参考文献DOI": "[]"
            }

    # ================================
    # Step 3. OpenAlex 抓取 + 主题解析
    # ================================
    def get_openalex_topic(self, doi: str) -> dict:
        """获取 OpenAlex 学科信息（topics + primary_topic）"""
        base = "https://api.openalex.org/works/"
        url = f"{base}https://doi.org/{doi}"
        headers = {"User-Agent": f"OpenAlex-Client (mailto:{self.openalex_email})"}

        for attempt in range(3):
            try:
                time.sleep(random.uniform(0.3, 0.7))
                r = requests.get(url, headers=headers, timeout=20)

                if r.status_code == 429:
                    wait = 3 * (attempt + 1)
                    time.sleep(wait)
                    continue
                if r.status_code == 404:
                    return {"OpenAlex_field_subfield_list": []}

                r.raise_for_status()
                data = r.json()

                topics = data.get("topics") or []
                primary = data.get("primary_topic") or None
                all_fields = set()

                def get_name(obj: Any):
                    return obj.get("display_name") if isinstance(obj, dict) else str(obj)

                for t in topics:
                    f, s = get_name(t.get("field")), get_name(t.get("subfield"))
                    if f:
                        all_fields.add(f)
                    if s:
                        all_fields.add(s)

                if primary:
                    f, s = get_name(primary.get("field")), get_name(primary.get("subfield"))
                    if f:
                        all_fields.add(f)
                    if s:
                        all_fields.add(s)

                return {"OpenAlex_field_subfield_list": list(all_fields)}

            except Exception:
                time.sleep(2)
        return {"OpenAlex_field_subfield_list": []}

    # ================================
    # Step 4. 合并 + 输出
    # ================================
    def merge_metadata_with_crossref(self, limit: int = 500) -> str:
        if self.df_raw is None:
            self.load_all_csvs()

        doi_col = [col for col in self.df_raw.columns if col.strip().lower() == "doi"][0]
        records = []

        print(f"🔍 开始获取 Crossref 元数据 + OpenAlex 学科主题（limit={limit}）...")

        def safe_fetch(doi: str) -> Dict[str, Any]:
            record = self.get_crossref_metadata(doi)
            openalex_data = self.get_openalex_topic(doi)
            record["OpenAlex_field_subfield_list"] = openalex_data["OpenAlex_field_subfield_list"]
            return record

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(safe_fetch, doi): doi for doi in self.df_raw[doi_col]}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching Data", ncols=90):
                try:
                    records.append(future.result())
                except Exception as e:
                    records.append({
                        "DOI": futures[future],
                        "CR_摘要": "",
                        "CR_出版商": "",
                        "CR_作者和机构": "[]",
                        "CR_参考文献DOI": "[]",
                        "OpenAlex_field_subfield_list": [],
                        "OpenAlex_map_subjects": []
                    })

        df_crossref = pd.DataFrame(records)
        self.df_merged = pd.merge(self.df_raw, df_crossref, left_on=doi_col, right_on="DOI", how="left")
        self.df_merged = self.df_merged[self.df_merged["CR_摘要"].fillna("").str.strip() != ""]
        total_valid = len(self.df_merged)

        # ========= 加载映射表 =========
        map_path = self.map_path
        openalex_to_cn = {}
        if map_path.exists():
            with open(map_path, "r", encoding="utf-8") as f:
                openalex_to_cn = json.load(f)
            print(f"✅ 已加载映射表：{map_path}")
        else:
            print(f"⚠️ 未找到映射文件: {map_path}")

        def map_to_cn_subjects(subfields: List[str]) -> List[str]:
            cn_subjects = set()
            for s in subfields or []:
                mapped = openalex_to_cn.get(s)
                if mapped:
                    cn_subjects.update(mapped if isinstance(mapped, list) else [mapped])
            return sorted(cn_subjects)

        self.df_merged["OpenAlex_map_subjects"] = self.df_merged["OpenAlex_field_subfield_list"].apply(map_to_cn_subjects)

        # ========= 删除空行逻辑 =========
        def is_empty_list_or_none(x):
            if x is None:
                return True
            if isinstance(x, str):
                x = x.strip()
                if x in ("", "[]", "nan", "NaN", "None"):
                    return True
            if isinstance(x, (list, tuple, set)):
                return len(x) == 0
            return False

        before = len(self.df_merged)
        self.df_merged = self.df_merged[
            ~(
                self.df_merged["OpenAlex_field_subfield_list"].apply(is_empty_list_or_none)
                | self.df_merged["OpenAlex_map_subjects"].apply(is_empty_list_or_none)
            )
        ].reset_index(drop=True)
        after = len(self.df_merged)
        print(f"🧹 已删除 {before - after} 条 OpenAlex 为空的记录，剩余 {after} 条有效数据。")

        # ========= 限制数量与保存 =========
        if after > limit:
            self.df_merged = self.df_merged.head(limit)

        os.makedirs(self.output_dir, exist_ok=True)
        file_name = os.path.basename(self.input_dir.rstrip("/")) + ".csv"
        out_path = os.path.join(self.output_dir, file_name)
        self.df_merged.to_csv(out_path, index=False, encoding="utf-8-sig")

        print(f"✅ 已保存到: {out_path} | 有效论文数: {len(self.df_merged)}")
        return out_path

    # ================================
    # Step 5. 统计信息
    # ================================
    def print_statistics(self):
        if self.df_merged is None or len(self.df_merged) == 0:
            print("⚠️ 无有效记录。")
            return
        total = len(self.df_merged)
        no_abs = (self.df_merged["CR_摘要"] == "").sum()
        print("📊 统计信息")
        print(f"- 总论文数: {total}")
        print(f"- 缺失摘要: {no_abs} ({no_abs/total:.1%})")


if __name__ == "__main__":
    processor = CrossrefMetaProcessor("data/01meta_data/0101 Philosophy")  # 输入目录
    processor.merge_metadata_with_crossref(limit=500)
    processor.print_statistics()

# -*- coding: utf-8 -*-
# Created by Messimeimei

"""
可增量补充扩充数据的脚本：基于 Crossref 和 OpenAlex 抓取元数据
    1. 对于当前下载的数据（data/01meta_data/<学科>），
       对比原始数据（data/01origin_meta_data/<学科>），求当前数据的新增 DOI 列表
    2. 对新增 DOI 列表，抓取 Crossref 和 OpenAlex 元数据
    3. 目标路径是 data/02crossref_data/<学科>.csv，将新增 DOI 的完整数据追加进去，去重，覆盖保存
"""

import random
import time
import os
import re
import json
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any


class CrossrefMetaProcessor:
    """
    这个类完成数据集构建的第一阶段：从01meta_data目录获取原始数据，
    并通过Crossref和OpenAlex API抓取补充元数据，最后保存到02crossref_data对应学科csv文件中
    """
    def __init__(self, input_dir: str,
                 output_dir: str = "data/02crossref_data",
                 openalex_email: str = "3170529323@qq.com",
                 origin_dir: str = None):     # 传入 01origin_meta_data 的学科目录，用于增量计算

            ROOT_DIR = Path(__file__).resolve().parents[2]
            self.input_dir = str((ROOT_DIR / input_dir).resolve())          # 01meta_data/<学科>
            self.output_dir = str((ROOT_DIR / output_dir).resolve())        # 02crossref_data
            self.map_path = ROOT_DIR / "data/deepseek_map.json"    # 映射文件路径
            self.openalex_email = openalex_email
            self.df_raw = None
            self.df_merged = None
            self.origin_dir = str((ROOT_DIR / origin_dir).resolve()) if origin_dir else None


    # ================================ 工具函数 ================================
    @staticmethod
    def _strip_tags(text: str) -> str:
        # 取出任意HTML/XML标签，并将连续空白替换为单个空格
        if not text:
            return ""
        txt = re.sub(r"<[^>]+>", "", text)
        txt = re.sub(r"\s+", " ", txt)
        return txt.strip()

    # ================================ Step 1. df格式返回清洗后的 CSV ================================
    def load_all_csvs(self) -> pd.DataFrame:
        """
        加载学科目录下所有 CSV 文件（这里实际上就只有一个CSV文件）
        进行清洗，只保留 "DOI", "来源", "研究方向", "论文标题" 4列，
        并合并为一个 DataFrame 返回
        """
        all_dfs = []
        for file in os.listdir(self.input_dir):
            if not file.endswith(".csv"):
                continue
            try:
                df = pd.read_csv(os.path.join(self.input_dir, file), encoding="utf-8-sig")
                all_dfs.append(df)
            except Exception as e:
                print(f"⚠️ 文件 {file} 读取失败: {e}")

        if not all_dfs:
            raise RuntimeError("❌ 输入目录下没有可用的 CSV 文件")
        
        # 合并所有的 DataFrame，实际上这里只有一个
        df = pd.concat(all_dfs, ignore_index=True)

        # ---------- 有时候下载中文名，有时候下载英文名，定义2种文件中的csv字段名称 ----------
        col_alias = {
            "doi": ["doi", "DOI"],
            "source": ["source", "来源", "Source"],
            "field": ["research area", "研究方向", "Research Area"],
            "title": ["article title", "论文标题", "Article Title"],
        }

        cols_lower = [c.strip().lower() for c in df.columns]

        def find_col(keys):
            """不论是中文还是英文列，都可以找到"""
            for k in keys:
                for c in df.columns:
                    if c.strip().lower() == k.lower():
                        return c
            return None

        doi_col = find_col(col_alias["doi"])
        source_col = find_col(col_alias["source"])
        field_col = find_col(col_alias["field"])
        title_col = find_col(col_alias["title"])

        # 判断文件是否缺少某个列
        missing = [k for k, v in {
            "DOI": doi_col,
            "Source/来源": source_col,
            "Research Area/研究方向": field_col,
            "Article Title/论文标题": title_col,
        }.items() if v is None]

        if missing:
            raise RuntimeError(f"❌ 缺少必要列: {missing}\n当前列名: {list(df.columns)}")

        # 清洗 DOI ，去重，存在/，去空 
        df[doi_col] = df[doi_col].astype(str).str.strip()
        df = df.dropna(subset=[doi_col])
        df = df[df[doi_col].str.contains("/", na=False)]
        df = df.drop_duplicates(subset=[doi_col]).reset_index(drop=True)

        # 数据清洗，只保留原始数据的4个列并统一成4个字段
        self.df_raw = df[[doi_col, source_col, field_col, title_col]]
        self.df_raw.columns = ["DOI", "来源", "研究方向", "论文标题"]  # 统一中文表头
        return self.df_raw

    # ================================ Step 2. Crossref 抓取 ================================
    @staticmethod
    def get_crossref_metadata(doi: str) -> dict:
        """
        输入单篇论文 DOI，返回以 DOI 为键，值为包含摘要、作者机构、参考文献DOI的字典
        """
        base_url = "https://api.crossref.org/works/"
        url = f"{base_url}{doi}"
        try:
            headers = {"User-Agent": "Mozilla/5.0 (mailto:3170529323@qq.com)"}
            response = requests.get(url, headers=headers, timeout=25)
            response.raise_for_status()
            msg = response.json().get("message", {})

            abstract_txt = CrossrefMetaProcessor._strip_tags(msg.get("abstract", ""))

            # 作者及机构
            author_info = []
            for a in msg.get("author", []) or []:
                name = (a.get("given", "") + " " + a.get("family", "")).strip()
                affs = [aff.get("name") for aff in a.get("affiliation", []) if aff.get("name")]
                if name or affs:
                    author_info.append({"name": name, "affiliation": affs})

            # 参考文献 DOI
            ref_dois = []
            for ref in msg.get("reference", []) or []:
                doi_ref = ref.get("DOI") or ref.get("doi")
                if doi_ref:
                    ref_dois.append(doi_ref.strip())

            return {
                "DOI": doi,
                "CR_摘要": abstract_txt,
                "CR_作者和机构": json.dumps(author_info, ensure_ascii=False),
                "CR_参考文献DOI": json.dumps(ref_dois, ensure_ascii=False)
            }

        except Exception:
            return {
                "DOI": doi,
                "CR_摘要": "",
                "CR_作者和机构": "[]",
                "CR_参考文献DOI": "[]"
            }

    # ================================ Step 3. OpenAlex 抓取 ================================
    def get_openalex_topic(self, doi: str) -> dict:
        """
        输入单篇论文 DOI，返回以 OpenAlex_field_list 和 OpenAlex_subfield_list 为键的字典
        """
        base = "https://api.openalex.org/works/"
        url = f"{base}https://doi.org/{doi}"
        headers = {"User-Agent": f"OpenAlex-Client (mailto:{self.openalex_email})"}

        # 单篇论文最多尝试3次，否则返回空值
        for attempt in range(3):
            try:
                time.sleep(random.uniform(0.3, 0.7))
                r = requests.get(url, headers=headers, timeout=20)
                
                if r.status_code == 429:
                    time.sleep(3 * (attempt + 1))
                    continue
                if r.status_code == 404:
                    return {"OpenAlex_field_list": [], "OpenAlex_subfield_list": []}

                r.raise_for_status()
                data = r.json()

                topics = data.get("topics") or []
                primary = data.get("primary_topic") or None

                field_set, subfield_set = set(), set()
                def get_name(obj: Any):
                    return obj.get("display_name") if isinstance(obj, dict) else str(obj)

                for t in topics:
                    f, s = get_name(t.get("field")), get_name(t.get("subfield"))
                    if f: field_set.add(f)
                    if s: subfield_set.add(s)

                if primary:
                    f, s = get_name(primary.get("field")), get_name(primary.get("subfield"))
                    if f: field_set.add(f)
                    if s: subfield_set.add(s)

                return {
                    "OpenAlex_field_list": list(field_set),
                    "OpenAlex_subfield_list": list(subfield_set)
                }

            except Exception:
                time.sleep(2)
        return {"OpenAlex_field_list": [], "OpenAlex_subfield_list": []}

    # ================ Step 4. 合并 + 增量更新（基于原始数据与当前数据的差集）=============
    def merge_metadata_with_crossref(self, limit: int = 500):
        """
        总流程：
        1. 读取当前数据和原始数据，计算 DOI 差集，得到新增 DOI 列表
        2. 对新增 DOI 列表，抓取 Crossref 和 OpenAlex 元数据
        3. 合并新增 DOI 的元数据与当前数据的其他列，得到新增 DOI 的完整数据
        4. 追加到目标文件（02crossref_data/<学科>.csv）→ 去重 → 覆盖保存
        """

        # 读取当前的数据（有可能比原始数据有所增加），并转换成df格式， ["DOI","来源","研究方向","论文标题"]
        if self.df_raw is None:
            self.load_all_csvs()

        if not self.origin_dir or not os.path.isdir(self.origin_dir):
            raise RuntimeError(f"❌ 未提供有效的 origin_dir（01origin_meta_data 学科目录）：{self.origin_dir}")

        # 读取原始的数据，并统计其数量
        origin_csvs = [f for f in os.listdir(self.origin_dir) if f.lower().endswith(".csv")]
        if len(origin_csvs) == 0:
            raise RuntimeError(f"❌ {self.origin_dir} 下没有 CSV 文件")
        if len(origin_csvs) > 1:
            raise RuntimeError(f"❌ {self.origin_dir} 下有 {len(origin_csvs)} 个 CSV，期望仅 1 个")
        origin_csv = os.path.join(self.origin_dir, origin_csvs[0])

        def _read_csv_any(path):
            # 返回原始数据的df格式
            for enc in ("utf-8-sig", "utf-8", "gb18030"):
                try:
                    return pd.read_csv(path, encoding=enc)
                except Exception:
                    pass
            raise RuntimeError(f"❌ 文件读取失败：{path}")

        df_origin_raw = _read_csv_any(origin_csv)

        # 复用原先的列名识别逻辑
        col_alias = {
            "doi": ["doi", "DOI"],
            "source": ["source", "来源", "Source"],
            "field": ["research area", "研究方向", "Research Area"],
            "title": ["article title", "论文标题", "Article Title"],
        }

        def _find_col(df, keys):
            for k in keys:
                for c in df.columns:
                    if c.strip().lower() == k.lower():
                        return c
            return None

        # 分别获取当前数据和原始数据的 DOI 列
        doi_col_01 = [c for c in self.df_raw.columns if c.strip().lower() == "doi"][0]
        doi_col_00 = _find_col(df_origin_raw, col_alias["doi"])
        if doi_col_00 is None:
            raise RuntimeError(f"❌ 00目录 CSV 缺少 DOI 列：{origin_csv}\n列名: {list(df_origin_raw.columns)}")

        # 清洗原始数据的 DOI
        df_origin_raw[doi_col_00] = df_origin_raw[doi_col_00].astype(str).str.strip()
        df_origin_raw = df_origin_raw.dropna(subset=[doi_col_00])
        df_origin_raw = df_origin_raw[df_origin_raw[doi_col_00].str.contains("/", na=False)]
        df_origin_raw = df_origin_raw.drop_duplicates(subset=[doi_col_00]).reset_index(drop=True)

        # 求新增的数据的 DOI 集合
        dois_01 = set(self.df_raw[doi_col_01].astype(str))
        dois_00 = set(df_origin_raw[doi_col_00].astype(str))

        new_dois = sorted(list(dois_01 - dois_00))
        if len(new_dois) == 0:
            # 即便新增为 0，也要保证目标文件存在；若已存在则不动，若不存在可以直接返回
            file_name = os.path.basename(self.input_dir.rstrip("/")) + ".csv"
            out_path = os.path.join(self.output_dir, file_name)
            print("✅ 新增 DOI 数量为 0，跳过抓取。")
            return out_path

        # 限制对于新增的数据，只抓取前 limit 条
        if len(new_dois) > limit:
            new_dois = new_dois[:limit]
        print(f"🔍 基于 01-00 差集，需抓取新增 DOI 数量：{len(new_dois)}")

        # 从当前数据中（df_raw）获取其他的元数据（来源/研究方向/标题等）
        df_new_input = self.df_raw[self.df_raw[doi_col_01].isin(new_dois)].reset_index(drop=True)

        # ======== 对新增数据进行 Openalxe 和 CrossRef 数据的补充 ========
        from concurrent.futures import ThreadPoolExecutor, as_completed  # 避免顶部重复导入报错
        records = []

        def safe_fetch(doi: str) -> Dict[str, Any]:
            # 对新增的 DOI 进行抓取
            record = self.get_crossref_metadata(doi)
            openalex_data = self.get_openalex_topic(doi)
            record.update(openalex_data)
            return record

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(safe_fetch, doi): doi for doi in df_new_input[doi_col_01]}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching Data", ncols=90):
                try:
                    records.append(future.result())
                except Exception:
                    records.append({
                        "DOI": futures[future],
                        "CR_摘要": "",
                        "CR_作者和机构": "[]",
                        "CR_参考文献DOI": "[]",
                        "OpenAlex_field_list": [],
                        "OpenAlex_subfield_list": [],
                        "OpenAlex_map_subjects": []
                    })

        # 合并新增 DOI 的抓取结果与新增 DOI 的其他列，得到最终的新增 DOI 的完整数据
        df_crossref = pd.DataFrame(records)
        self.df_merged = pd.merge(df_new_input, df_crossref, left_on=doi_col_01, right_on="DOI", how="left")
        self.df_merged = self.df_merged[self.df_merged["CR_摘要"].fillna("").str.strip() != ""]

        # 映射表加载
        openalex_to_cn = {}
        if self.map_path.exists():
            with open(self.map_path, "r", encoding="utf-8") as f:
                openalex_to_cn = json.load(f)
            print(f"✅ 已加载映射表：{self.map_path}")
        else:
            print(f"⚠️ 未找到映射文件: {self.map_path}")

        # 将 OpenAlex 的 field 与 subfield 2列映射为中国学科和对应的分数
        def map_to_cn_groups(fields: List[str], subfields: List[str]) -> List[List[tuple]]:
            groups = []
            for name in (fields or []) + (subfields or []):
                mapped_pairs = openalex_to_cn.get(name, [])
                clean_pairs = []
                for m in mapped_pairs:
                    if isinstance(m, (list, tuple)) and len(m) == 2:
                        subj, score = m
                        try:
                            clean_pairs.append((str(subj), float(score)))
                        except Exception:
                            continue
                groups.append(clean_pairs)
            return groups

        self.df_merged["OpenAlex_map_subjects"] = self.df_merged.apply(
            lambda r: map_to_cn_groups(r["OpenAlex_field_list"], r["OpenAlex_subfield_list"]),
            axis=1
        )

        # 去掉 OpenAlex_field_list 和 OpenAlex_subfield_list 都为空的记录
        def is_empty_list_or_none(x):
            if x is None:
                return True
            if isinstance(x, str) and x.strip() in ("", "[]", "nan", "None"):
                return True
            if isinstance(x, (list, tuple, set)) and len(x) == 0:
                return True
            return False

        before = len(self.df_merged)
        self.df_merged = self.df_merged[
            ~(
                self.df_merged["OpenAlex_field_list"].apply(is_empty_list_or_none)
                & self.df_merged["OpenAlex_subfield_list"].apply(is_empty_list_or_none)
            )
        ].reset_index(drop=True)
        after = len(self.df_merged)
        print(f"🧹 过滤无学科数据: {before - after} 条，剩余 {after} 条。")

        # ======== 追加到目标文件（02crossref_data/<学科>.csv）→ 去重 → 覆盖保存 ========
        file_name = os.path.basename(self.input_dir.rstrip("/")) + ".csv"
        out_path = os.path.join(self.output_dir, file_name)

        # 读取目标文件的旧数据（如果存在），并把新增的 DOI 完整数据追加进去，去重
        df_old = None
        if os.path.exists(out_path):
            try:
                df_old = pd.read_csv(out_path)
                print(f"🔁 检测到目标历史数据: {len(df_old)} 条，将执行追加并去重...")
            except Exception as e:
                print(f"⚠️ 读取旧文件失败，将直接创建新文件: {e}")

        if df_old is not None:
            # 确保新增 DOI 数据的列与旧数据列一致，并将新的 DOI 完整数据追加进去
            for col in self.df_merged.columns:
                if col not in df_old.columns:
                    df_old[col] = None
            for col in df_old.columns:
                if col not in self.df_merged.columns:
                    self.df_merged[col] = None
            df_all = pd.concat([df_old, self.df_merged], ignore_index=True)
        else:
            df_all = self.df_merged

        # 按照 DOI 去重并保存到原始文件中
        df_all = df_all.drop_duplicates(subset=["DOI"]).reset_index(drop=True)
        os.makedirs(self.output_dir, exist_ok=True)
        df_all.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"✅ 已保存: {out_path} | 总记录数: {len(df_all)}")

        return out_path


    # ================================ Step 5. 统计 ================================
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
    processor = CrossrefMetaProcessor(
        input_dir="data/01meta_data/0101 Philosophy",
        origin_dir="data/01origin_meta_data/0101 Philosophy",   # ← 00目录同名学科
        output_dir="data/02crossref_data"
    )
    processor.merge_metadata_with_crossref(limit=8000)
    processor.print_statistics()

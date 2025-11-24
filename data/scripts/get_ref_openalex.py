# -*- coding: utf-8 -*-
# Created by Messimeimei
"""
增量操作，为所有的参考文献获取 OpenAlex 学科并映射至中国一级学科，在上一步的csv文件基础上新增2列：
- Ref_OpenAlex_topics：参考文献的 OpenAlex 学科主题列表，格式为[[[field1, field2], [subfield1]], ...]
- Ref_OpenAlex_map_subjects：参考文献的映射中国一级学科列表，格式为[[[subject1, score1], [subject2, score2]], ...]
数据位于data/03openalex_data目录下，文件名与输入文件相同
"""

import os
import json
import time
import random
import requests
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Dict, Any


class RefOpenAlexMapper:
    """
    这个类完成数据集构建的第二阶段：从02crossref_data目录获取原始数据，
    并通过OpenAlex API为参考文献的所有论文抓取补充元数据，最后保存到03openalex_data对应学科csv文件中
    """
    def __init__(self,
                 input_file: str,
                 origin_file: str = None,
                 output_dir: str = "data/03openalex_data",
                 openalex_email: str = "3170529323@qq.com"):

        ROOT_DIR = Path(__file__).resolve().parents[2]
        self.input_file = (ROOT_DIR / input_file).resolve()

        if origin_file:
            origin_path = (ROOT_DIR / origin_file).resolve()
            if origin_path.is_dir():
                expected = origin_path / self.input_file.name
                if expected.exists():
                    print(f"🔗 基线文件匹配成功: {expected.name}")
                    origin_path = expected
                else:
                    print(f"⚠️ 未找到对应基线文件: {expected.name}，视为首次运行（全量处理）")
                    origin_path = None
            self.origin_file = origin_path
        else:
            self.origin_file = None

        self.output_dir = (ROOT_DIR / output_dir).resolve()
        self.map_path = ROOT_DIR / "data/deepseek_map.json"
        self.openalex_email = openalex_email

        os.makedirs(self.output_dir, exist_ok=True)
        self.df_mapped = None

    # ================== Step 1：获取新增论文数据，以 DOI ==================
    def load_incremental_data(self) -> pd.DataFrame:
        """
        读取当前数据csv文件和原始数据的csv文件，对比求新增 DOI 差集
        """

        # 读取当前数据，df格式
        abs_input = str(self.input_file.resolve())
        print(f"\n📘 加载最新数据: {self.input_file.name}")
        print(f"    ↳ 路径: {abs_input}")
        df_new = pd.read_csv(self.input_file, encoding="utf-8-sig")

        # 如果没有原始数据，则对当前数据所有论文执行操作
        if not self.origin_file or not Path(self.origin_file).exists():
            print("⚠️ 无基线数据，首次执行 → 全量处理")
            df_new["is_new"] = True
            return df_new

        # 读取原始数据，df格式
        abs_origin = str(Path(self.origin_file).resolve())
        print(f"📘 加载基线数据: {Path(self.origin_file).name}")
        print(f"    ↳ 路径: {abs_origin}")
        df_old = pd.read_csv(self.origin_file, encoding="utf-8-sig")
    
        # 计算新增 DOI 差集
        new_doi_set = set(df_new["DOI"].astype(str))
        old_doi_set = set(df_old["DOI"].astype(str))
        added_dois = new_doi_set - old_doi_set

        print(f"🔍 当前 {len(new_doi_set)} 条，基线 {len(old_doi_set)} 条 → 新增 {len(added_dois)} 条")
        df_new["is_new"] = df_new["DOI"].astype(str).isin(added_dois)

        return df_new[df_new["is_new"]].reset_index(drop=True)

    # ================== Step 2：为参考文献获取 Openalex 数据 ==================
    def safe_request(self, url, headers, retries=3):
        # 为单篇参考文献获取 OpenAlex 数据，最多重试 retries 次，否则返回空
        for attempt in range(retries):
            try:
                time.sleep(random.uniform(0.6, 1.2))
                r = requests.get(url, headers=headers, timeout=25)
                if r.status_code == 429:
                    wait = 5 * (attempt + 1)
                    print(f"⚠️ 429 Too Many Requests，等待 {wait}s...")
                    time.sleep(wait)
                    continue
                if r.status_code == 404:
                    return None
                r.raise_for_status()
                return r.json()
            except Exception:
                time.sleep(2)
        return None

    def get_openalex_topics(self, doi: str) -> Dict[str, List[str]]:
        # 具体执行为单篇参考文献获取 OpenAlex 学科主题的操作

        url = f"https://api.openalex.org/works/https://doi.org/{doi}"
        headers = {"User-Agent": f"OpenAlex-Client (mailto:{self.openalex_email})"}
        data = self.safe_request(url, headers)

        if not data:
            return {"fields": [], "subfields": []}

        # 解析单篇论文的 OpenAlex 学科主题
        topics = data.get("topics") or []
        primary = data.get("primary_topic") or None
        fields, subfields = set(), set()
        get_name = lambda o: o.get("display_name") if isinstance(o, dict) else str(o)

        for t in topics:
            f, s = get_name(t.get("field")), get_name(t.get("subfield"))
            if f: fields.add(f)
            if s: subfields.add(s)

        if primary:
            f, s = get_name(primary.get("field")), get_name(primary.get("subfield"))
            if f: fields.add(f)
            if s: subfields.add(s)

        return {"fields": list(fields), "subfields": list(subfields)}

    # ================== Step 3：中国学科映射 ==================
    def load_mapping_table(self):
        # 加载中国学科映射表
        if not self.map_path.exists():
            raise RuntimeError(f"❌ 映射表缺失: {self.map_path}")
        with open(self.map_path, "r", encoding="utf-8") as f:
            print(f"✅ 映射表已加载: {self.map_path}")
            return json.load(f)

    # ================== Step 4：主执行流程 ==================
    def process_ref_openalex(self, max_ref_per_paper=20, max_workers=8, max_rows=None):
        """
        为新增的论文参考文献获取 OpenAlex 学科主题并映射至中国一级学科
        """
        
        df_new = self.load_incremental_data()
        if df_new.empty:
            print("✅ 无新增论文，跳过执行")
            return None

        mapping_dict = self.load_mapping_table()

        # 限制处理条数，也是方便测试，默认不限制全部处理新增论文
        if max_rows and len(df_new) > max_rows:
            print(f"⚠️ 新增 {len(df_new)} 条，仅处理前 {max_rows} 条")
            df_new = df_new.head(max_rows)

        print(f"🔍 开始抓取参考文献学科映射（新增 {len(df_new)} 条）...")

        topics_results = [None] * len(df_new)
        mapped_results = [None] * len(df_new)
        success_count = 0

        # 并发抓取参考文献的 OpenAlex 学科主题并映射
        def fetch(ref_dois):
            ref_topics, ref_cn = [], []
            for ref_doi in ref_dois[:max_ref_per_paper]:
                # 获取单篇参考文献的 OpenAlex 学科主题
                topic = self.get_openalex_topics(ref_doi)
                if not topic["fields"] and not topic["subfields"]:
                    continue
                ref_topics.append([topic["fields"], topic["subfields"]])

                # 为单篇参考文献的学科主题映射中国一级学科
                mapped = []
                for name in topic["fields"] + topic["subfields"]:
                    for subj, score in mapping_dict.get(name, []):
                        mapped.append([subj, float(score)])
                
                # 保存单篇参考文献的原始结果和映射结果
                ref_cn.append(mapped)
            return ref_topics, ref_cn

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for idx, row in df_new.iterrows():
                try:
                    ref_dois = json.loads(row["CR_参考文献DOI"])
                    if ref_dois:
                        futures[executor.submit(fetch, ref_dois)] = idx
                except:
                    pass

            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="Fetching Reference Topics", ncols=90):
                idx = futures[future]
                try:
                    topics, mapped = future.result()
                    if mapped: success_count += 1
                    topics_results[idx] = topics
                    mapped_results[idx] = mapped
                except Exception:
                    topics_results[idx] = []
                    mapped_results[idx] = []

        # 为新增论文添加2列结果
        df_new["Ref_OpenAlex_topics"] = topics_results
        df_new["Ref_OpenAlex_map_subjects"] = mapped_results

        # 过滤学科为空的结果，删除is_new临时列
        df_new = df_new[df_new["Ref_OpenAlex_map_subjects"].apply(lambda x: isinstance(x, list) and len(x) > 0)]
        df_new = df_new.drop(columns=["is_new"], errors="ignore")

        print(f"✅ 新增论文 {len(futures)} 条，其中成功获取 OpenAlex 数据 {success_count} 条")

        # 将新增数据与目标文件内容合并
        out_path = (self.output_dir / self.input_file.name).resolve()
        print(f"\n💾 输出目标文件: {out_path}")

        if out_path.exists():
            df_old = pd.read_csv(out_path, encoding="utf-8-sig")
            old_dois = set(df_old["DOI"].astype(str))
            df_append = df_new[~df_new["DOI"].astype(str).isin(old_dois)]
            df_all = pd.concat([df_old, df_append], ignore_index=True)
            print(f"🧩 原文件 {len(df_old)} 条，追加 {len(df_append)} 条 → 合计 {len(df_all)} 条")
        else:
            df_all = df_new

        # 保存加入新增数据后的完整文件
        df_all.to_csv(out_path, index=False, encoding="utf-8-sig")
        self.df_mapped = df_all
        print(f"✅ 已保存结果 → {out_path}")

    # ================== Step 5：统计 ==================
    def print_statistics(self):
        if self.df_mapped is None:
            print("⚠️ 无结果可统计")
            return
        total = len(self.df_mapped)
        valid = 0
        for v in self.df_mapped["Ref_OpenAlex_map_subjects"]:
            try:
                if isinstance(v, list) and len(v) > 0:
                    valid += 1
            except:
                pass
        print(f"\n📊 统计: 总论文 {total} | 成功映射 {valid} ({valid/total:.1%})")



# ================== 单文件测试入口 ==================
if __name__ == "__main__":
    mapper = RefOpenAlexMapper(
        input_file="data/02crossref_data/0802 Mechanical Engineering.csv",
        origin_file="data/02origin_crossref_data",
        output_dir="data/03openalex_data",
    )
    mapper.process_ref_openalex(max_ref_per_paper=5, max_workers=5)
    mapper.print_statistics()

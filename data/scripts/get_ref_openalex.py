# -*- coding: utf-8 -*-
# Created by Messimeimei
# Updated by ChatGPT (2025/10)
"""
并发抓取每篇论文参考文献的 OpenAlex 学科并映射为中国一级学科
输入：data/02crossref_data/<文件>.csv
输出：data/03openalex_data/<同名>.csv
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
    def __init__(self,
                 input_file: str,
                 output_dir: str = "data/03openalex_data",
                 openalex_email: str = "3170529323@qq.com"):
        """初始化路径"""
        ROOT_DIR = Path(__file__).resolve().parents[2]
        self.input_file = (ROOT_DIR / input_file).resolve()
        self.output_dir = (ROOT_DIR / output_dir).resolve()
        self.map_path = ROOT_DIR / "data/openalex_to_cn_disciplines_merged.json"
        self.openalex_email = openalex_email

        os.makedirs(self.output_dir, exist_ok=True)
        self.df_raw = None
        self.df_mapped = None

    # ================================
    # Step 1. 加载输入数据
    # ================================
    def load_csv(self) -> pd.DataFrame:
        """读取单个 CSV 文件"""
        print(f"📘 正在读取文件: {self.input_file.name}")
        try:
            df = pd.read_csv(self.input_file, encoding="utf-8-sig")
            if "CR_参考文献DOI" not in df.columns:
                raise KeyError("文件缺少 'CR_参考文献DOI' 列")
            self.df_raw = df
            return df
        except Exception as e:
            raise RuntimeError(f"❌ 无法读取文件 {self.input_file}: {e}")

    # ================================
    # Step 2. OpenAlex 获取函数
    # ================================
    def safe_request(self, url, headers, retries=3):
        for attempt in range(retries):
            try:
                time.sleep(random.uniform(0.3, 0.8))
                r = requests.get(url, headers=headers, timeout=25)
                if r.status_code == 429:
                    wait = 3 * (attempt + 1)
                    print(f"⚠️ 429 Too Many Requests，等待 {wait}s")
                    time.sleep(wait)
                    continue
                if r.status_code == 404:
                    return None
                r.raise_for_status()
                return r.json()
            except Exception:
                time.sleep(2)
        return None

    def get_openalex_topics(self, doi: str) -> List[str]:
        """获取单个 DOI 的 OpenAlex 学科列表"""
        base = "https://api.openalex.org/works/"
        url = f"{base}https://doi.org/{doi}"
        headers = {"User-Agent": f"OpenAlex-Client (mailto:{self.openalex_email})"}
        data = self.safe_request(url, headers)
        if not data:
            return []

        topics = data.get("topics") or []
        primary = data.get("primary_topic") or None
        all_fields = set()

        def get_name(obj):
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
        return list(all_fields)

    # ================================
    # Step 3. 加载映射表
    # ================================
    def load_mapping_table(self) -> Dict[str, List[str]]:
        """加载 OpenAlex→中国一级学科 映射"""
        if not self.map_path.exists():
            print(f"❌ 映射文件不存在: {self.map_path}")
            return {}
        with open(self.map_path, "r", encoding="utf-8") as f:
            print(f"✅ 已加载映射表: {self.map_path}")
            return json.load(f)

    @staticmethod
    def map_to_cn_subjects(subfields: List[str], mapping_dict: Dict[str, Any]) -> List[str]:
        """映射为中国一级学科"""
        cn_subjects = set()
        for s in subfields or []:
            mapped = mapping_dict.get(s)
            if mapped:
                cn_subjects.update(mapped if isinstance(mapped, list) else [mapped])
        return sorted(cn_subjects)

    # ================================
    # Step 4. 主逻辑
    # ================================
    def process_ref_openalex(self, max_ref_per_paper: int = 20, max_workers: int = 10) -> str:
        """主执行函数：对每篇论文的参考文献进行OpenAlex映射"""
        df = self.load_csv()
        mapping_dict = self.load_mapping_table()
        if not mapping_dict:
            raise RuntimeError("❌ 映射表为空，无法执行")

        print(f"🔍 开始抓取参考文献 OpenAlex 学科（max_ref_per_paper={max_ref_per_paper}）...")

        results = [None] * len(df)

        def fetch_ref_subjects(ref_dois: List[str]) -> List[List[str]]:
            """对每篇论文的参考文献抓取OpenAlex+映射"""
            ref_subjects = []
            for ref_doi in ref_dois[:max_ref_per_paper]:
                topics = self.get_openalex_topics(ref_doi)
                if not topics:
                    ref_subjects.append([])
                    continue
                cn_list = self.map_to_cn_subjects(topics, mapping_dict)
                ref_subjects.append(cn_list)
            return ref_subjects

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for idx, row in df.iterrows():
                try:
                    ref_dois = json.loads(row["CR_参考文献DOI"])
                    if not ref_dois:
                        continue
                    futures[executor.submit(fetch_ref_subjects, ref_dois)] = idx
                except Exception:
                    continue

            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="Fetching Reference Topics", ncols=90):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    results[idx] = []

        df["Ref_OpenAlex_map_subjects"] = results
        self.df_mapped = df

        # 保存结果
        out_path = self.output_dir / self.input_file.name
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"✅ 已保存结果到: {out_path}")
        return str(out_path)

    # ================================
    # Step 5. 简单统计
    # ================================
    def print_statistics(self):
        if self.df_mapped is None:
            print("⚠️ 未找到处理结果。")
            return
        total = len(self.df_mapped)
        valid = self.df_mapped["Ref_OpenAlex_map_subjects"].apply(lambda x: isinstance(x, list)).sum()
        print("📊 统计信息")
        print(f"- 总论文数: {total}")
        print(f"- 成功映射的论文: {valid} ({valid/total:.1%})")


# ================================
# 主程序入口（测试）
# ================================
if __name__ == "__main__":
    processor = RefOpenAlexMapper("data/02crossref_data/0101 Philosophy.csv")
    processor.process_ref_openalex(max_ref_per_paper=5, max_workers=5)
    processor.print_statistics()

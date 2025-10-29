# -*- coding: utf-8 -*-
# Created by Messimeimei
# Updated by ChatGPT (2025/10)
"""
并发抓取每篇论文参考文献的 OpenAlex 学科并映射为中国一级学科
输入：data/02crossref_data/<文件>.csv
输出：data/03openalex_data/<同名>.csv

更新说明：
1. 每个参考文献的 OpenAlex 学科字段（Ref_OpenAlex_topics）：
   → [ [field_list], [subfield_list] ]
2. 每个参考文献的映射结果字段（Ref_OpenAlex_map_subjects）：
   → [ [ [学科, 分数], [学科, 分数], ... ], ... ]
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

    # ================================ Step 1. 加载输入数据 ================================
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

    # ================================ Step 2. 请求 OpenAlex ================================
    def safe_request(self, url, headers, retries=3):
        """带重试机制的安全请求"""
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

    def get_openalex_topics(self, doi: str) -> Dict[str, List[str]]:
        """获取单个 DOI 的 OpenAlex 学科，区分 field / subfield"""
        base = "https://api.openalex.org/works/"
        url = f"{base}https://doi.org/{doi}"
        headers = {"User-Agent": f"OpenAlex-Client (mailto:{self.openalex_email})"}
        data = self.safe_request(url, headers)
        if not data:
            return {"fields": [], "subfields": []}

        topics = data.get("topics") or []
        primary = data.get("primary_topic") or None
        field_set, subfield_set = set(), set()

        def get_name(obj):
            return obj.get("display_name") if isinstance(obj, dict) else str(obj)

        for t in topics:
            f, s = get_name(t.get("field")), get_name(t.get("subfield"))
            if f:
                field_set.add(f)
            if s:
                subfield_set.add(s)

        if primary:
            f, s = get_name(primary.get("field")), get_name(primary.get("subfield"))
            if f:
                field_set.add(f)
            if s:
                subfield_set.add(s)

        return {"fields": list(field_set), "subfields": list(subfield_set)}

    # ================================ Step 3. 映射表与转换 ================================
    def load_mapping_table(self) -> Dict[str, Any]:
        """加载 OpenAlex→中国一级学科 映射表"""
        if not self.map_path.exists():
            print(f"❌ 映射文件不存在: {self.map_path}")
            return {}
        with open(self.map_path, "r", encoding="utf-8") as f:
            print(f"✅ 已加载映射表: {self.map_path}")
            return json.load(f)

    @staticmethod
    def map_to_cn_subjects(field_list: List[str], subfield_list: List[str],
                           mapping_dict: Dict[str, Any]) -> List[List]:
        """
        将 field/subfield 映射为中国学科（带分数）
        返回：[[学科, 分数], ...]
        """
        results = []
        for name in (field_list or []) + (subfield_list or []):
            mapped = mapping_dict.get(name, [])
            if isinstance(mapped, (list, tuple)):
                for m in mapped:
                    if isinstance(m, (list, tuple)) and len(m) == 2:
                        subj, score = m
                        try:
                            results.append([str(subj), float(score)])
                        except Exception:
                            continue
        # 去重 + 保留最高分
        unique_dict = {}
        for subj, score in results:
            unique_dict[subj] = max(unique_dict.get(subj, 0), score)
        results = sorted([[k, v] for k, v in unique_dict.items()], key=lambda x: x[1], reverse=True)
        return results

    # ================================ Step 4. 主执行逻辑 ================================
    def process_ref_openalex(
            self,
            max_ref_per_paper: int = 20,
            max_workers: int = 16,
            max_rows: int = 1000
    ) -> str:
        """
        主执行函数：对每篇论文的参考文献进行 OpenAlex 抓取与映射
        参数：
          - max_ref_per_paper：每篇论文最多处理的参考文献数量
          - max_workers：并发线程数
          - max_rows：最多处理多少篇论文（超过则截断）
        """
        df = self.load_csv()
        mapping_dict = self.load_mapping_table()
        if not mapping_dict:
            raise RuntimeError("❌ 映射表为空，无法执行")

        # ===== 根据参数限制行数 =====
        if max_rows is not None and len(df) > max_rows:
            print(f"⚠️ 文件包含 {len(df)} 条记录，仅处理前 {max_rows} 条。")
            df = df.head(max_rows)

        print(f"🔍 开始抓取参考文献 OpenAlex 学科（max_ref_per_paper={max_ref_per_paper}, max_rows={len(df)}）...")

        topics_results = [None] * len(df)
        mapped_results = [None] * len(df)

        def fetch_ref_info(ref_dois: List[str]):
            """抓取并映射一篇论文的所有参考文献"""
            ref_topics, ref_mapped = [], []
            for ref_doi in ref_dois[:max_ref_per_paper]:
                topic_data = self.get_openalex_topics(ref_doi)
                fields, subfields = topic_data["fields"], topic_data["subfields"]
                ref_topics.append([fields, subfields])

                # 映射为中国学科（带分数）
                cn_mapped = self.map_to_cn_subjects(fields, subfields, mapping_dict)
                ref_mapped.append(cn_mapped)
            return ref_topics, ref_mapped

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for idx, row in df.iterrows():
                try:
                    ref_dois = json.loads(row["CR_参考文献DOI"])
                    if not ref_dois:
                        continue
                    futures[executor.submit(fetch_ref_info, ref_dois)] = idx
                except Exception:
                    continue

            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="Fetching Reference Topics", ncols=90):
                idx = futures[future]
                try:
                    ref_topics, ref_mapped = future.result()
                    topics_results[idx] = ref_topics
                    mapped_results[idx] = ref_mapped
                except Exception:
                    topics_results[idx] = []
                    mapped_results[idx] = []

        # 保存结果列
        df["Ref_OpenAlex_topics"] = topics_results
        df["Ref_OpenAlex_map_subjects"] = mapped_results
        self.df_mapped = df

        out_path = self.output_dir / self.input_file.name
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"✅ 已保存结果到: {out_path}")
        return str(out_path)

    # ================================ Step 5. 简单统计 ================================
    def print_statistics(self):
        if self.df_mapped is None:
            print("⚠️ 未找到处理结果。")
            return
        total = len(self.df_mapped)
        valid = self.df_mapped["Ref_OpenAlex_map_subjects"].apply(lambda x: isinstance(x, list)).sum()
        print("📊 统计信息")
        print(f"- 总论文数: {total}")
        print(f"- 成功映射的论文: {valid} ({valid/total:.1%})")


# ================================ 主程序入口 ================================
if __name__ == "__main__":
    processor = RefOpenAlexMapper("data/02crossref_data/0101 Philosophy.csv")
    processor.process_ref_openalex(max_ref_per_paper=5, max_workers=5)
    processor.print_statistics()

# Created by Messimeimei
# Created at 2025/9/16

"""
    通过crossref的rest接口获取数据
    具体步骤如下：
    1. 从incites网站下载的原始数据csv中提取出每篇论文的doi
    2. 对doi通过crossref的rest接口获取论文的元数据
    3. 把2种数据进行拼接，筛选出有用的元数据并保存到processed_data文件夹中
"""

import os
import re
import requests
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any


class CrossrefMetaProcessor:
    def __init__(self, csv_path: str, output_dir: str = "../processed_data"):
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.df_raw = None
        self.df_merged = None

    @staticmethod
    def _strip_tags(text: str) -> str:
        """去掉 JATS/XML 标签，返回规整的纯文本"""
        if not text:
            return ""
        # 去标签
        txt = re.sub(r"<[^>]+>", "", text)
        # 合并换行和多余空格
        txt = re.sub(r"\s+", " ", txt)
        return txt.strip()

    def extract_clean_doi(self) -> pd.DataFrame:
        """读取原始 CSV，清洗 DOI，并返回只包含要保留字段的 DataFrame"""
        df = pd.read_csv(self.csv_path, encoding="utf-8-sig")

        # 统一小写比对
        cols = [c.strip().lower() for c in df.columns]

        doi_col = [df.columns[i] for i, c in enumerate(cols) if c == "doi"][0]
        source_col = [df.columns[i] for i, c in enumerate(cols) if "来源" in c][0]
        field_col = [df.columns[i] for i, c in enumerate(cols) if "研究方向" in c][0]
        title_col = [df.columns[i] for i, c in enumerate(cols) if "论文标题" in c][0]

        # 清洗 DOI
        df[doi_col] = df[doi_col].astype(str).str.strip()
        df[doi_col] = df[doi_col].replace(["", "nan", "NaN", "None", "NULL"], pd.NA)
        df = df.dropna(subset=[doi_col])
        df = df[df[doi_col].str.contains("/", na=False)]
        df = df.drop_duplicates(subset=[doi_col]).reset_index(drop=True)

        # 保留需要的几列
        self.df_raw = df[[doi_col, source_col, field_col, title_col]]
        return self.df_raw

    @staticmethod
    def get_crossref_metadata(doi: str) -> dict:
        """根据 DOI 获取 Crossref 精简元数据"""
        base_url = "https://api.crossref.org/works/"
        url = f"{base_url}{doi}"

        try:
            headers = {"User-Agent": "Mozilla/5.0 (mailto:your_email@example.com)"}
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            msg = response.json().get("message", {})

            # 1. 参考文献 DOI 列表
            ref_dois: List[str] = []
            for r in msg.get("reference", []) or []:
                if r.get("DOI"):
                    ref_dois.append(r["DOI"])

            # 2. 作者 (姓名, 研究机构) 列表
            author_info: List[tuple] = []
            for a in msg.get("author", []) or []:
                name = " ".join([a.get("given", ""), a.get("family", "")]).strip()
                affs = [af.get("name", "") for af in a.get("affiliation", []) if af.get("name")]
                if not affs:
                    author_info.append((name, ""))  # 无机构时留空
                else:
                    for aff in affs:
                        author_info.append((name, aff))

            # 3. 摘要去标签 + 规整化
            abstract_txt = CrossrefMetaProcessor._strip_tags(msg.get("abstract", ""))

            return {
                "DOI": doi,
                "CR_学科": msg.get("subject", []),
                "CR_摘要": abstract_txt,
                "CR_作者和机构": author_info,
                "CR_参考文献DOI": ref_dois,
                "CR_出版商": msg.get("publisher", ""),
            }
        except Exception as e:
            print(f"获取 DOI 元数据失败: {doi}, 错误: {e}")
            return {
                "DOI": doi,
                "CR_学科": [],
                "CR_摘要": "",
                "CR_作者和机构": [],
                "CR_参考文献DOI": [],
                "CR_出版商": "",
            }

    def merge_metadata_with_crossref(self) -> str:
        """合并原始 CSV 与 Crossref 精简字段并保存"""
        if self.df_raw is None:
            self.extract_clean_doi()

        doi_col = [col for col in self.df_raw.columns if col.strip().lower() == "doi"][0]

        crossref_records = []
        for doi in tqdm(self.df_raw[doi_col], desc="Fetching Crossref metadata"):
            meta = self.get_crossref_metadata(doi)
            crossref_records.append(meta)

        crossref_df = pd.DataFrame(crossref_records)

        self.df_merged = pd.merge(self.df_raw, crossref_df, left_on=doi_col, right_on="DOI", how="left")

        os.makedirs(self.output_dir, exist_ok=True)
        out_file = os.path.join(
            self.output_dir,
            os.path.basename(self.csv_path).replace(".csv", "_merged.csv")
        )
        self.df_merged.to_csv(out_file, index=False, encoding="utf-8-sig")

        print(f"✅ 合并完成，已保存到: {out_file}")
        return out_file

    def print_statistics(self):
        """打印统计信息"""
        if self.df_merged is None:
            print("⚠️ 请先运行 merge_metadata_with_crossref()")
            return

        total = len(self.df_merged)
        no_abs = (self.df_merged["CR_摘要"] == "").sum()
        no_auth = self.df_merged["CR_作者和机构"].apply(lambda x: len(x) == 0).sum()
        no_refs = self.df_merged["CR_参考文献DOI"].apply(lambda x: len(x) == 0).sum()

        print("📊 统计信息")
        print(f"- 总论文数: {total}")
        print(f"- 缺失摘要: {no_abs} ({no_abs/total:.1%})")
        print(f"- 缺失作者信息: {no_auth} ({no_auth/total:.1%})")
        print(f"- 缺失参考文献: {no_refs} ({no_refs/total:.1%})")


if __name__ == "__main__":
    processor = CrossrefMetaProcessor("../meta_data/1304 Fine Art.csv")
    processor.merge_metadata_with_crossref()
    processor.print_statistics()


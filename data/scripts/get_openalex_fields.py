# -*- coding: utf-8 -*-
"""
基于 OpenAlex Topics API 获取所有一级(field)与二级(subfield)学科名称
每行一个，仅名称，自动去重 + 排序 + 保存为 CSV
兼容分页，无403问题
"""

import requests
import pandas as pd
from time import sleep

def fetch_all_topics(per_page=200):
    """获取所有 topics 的原始数据"""
    base_url = "https://api.openalex.org/topics"
    params = {
        "per-page": per_page,
        "sort": "id",
        "mailto": "hujingxuan@whu.edu.cn"
    }
    headers = {
        "User-Agent": "WuhanUniversity-SubjectCross-Project (mailto:hujingxuan@whu.edu.cn)"
    }

    topics = []
    cursor = "*"
    while cursor:
        params["cursor"] = cursor
        r = requests.get(base_url, params=params, headers=headers, timeout=60)
        r.raise_for_status()
        data = r.json()
        topics.extend(data.get("results", []))
        cursor = data.get("meta", {}).get("next_cursor")
        print(f"已获取 {len(topics)} 条...")
        if not cursor:
            break
        sleep(0.3)
    print(f"✅ 共获取 {len(topics)} 个 topics。")
    return topics


def extract_fields_and_subfields(topics):
    """提取 field 与 subfield 名称"""
    fields = set()
    subfields = set()
    for t in topics:
        field = t.get("field", {}).get("display_name")
        subfield = t.get("subfield", {}).get("display_name")
        if field:
            fields.add(field.strip())
        if subfield:
            subfields.add(subfield.strip())
    return sorted(fields, key=str.lower), sorted(subfields, key=str.lower)


def save_to_csv(items, filename):
    """保存为 CSV，每行一个名称"""
    df = pd.DataFrame(items)
    df.to_csv(filename, index=False, header=False, encoding="utf-8-sig")
    print(f"💾 已保存 {len(items)} 条记录至 {filename}")


def main():
    topics = fetch_all_topics()
    fields, subfields = extract_fields_and_subfields(topics)
    save_to_csv(fields, "openalex_fields.csv")
    save_to_csv(subfields, "openalex_subfields.csv")


if __name__ == "__main__":
    main()

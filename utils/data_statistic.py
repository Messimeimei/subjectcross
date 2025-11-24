# -*- coding: utf-8 -*-
# Created by Messimeimei
# Updated by ChatGPT (2025/11)

import os
import json


def count_csv_rows_to_json(base_dir="../data", output_json="report_file_stats1.0.json"):
    target_dirs = [
        "01meta_data",
        "02crossref_data",
        "03openalex_data",
        "04input_data",
        "05output_data",
        "06finetune_data",
    ]

    report = {}

    for folder in target_dirs:
        folder_path = os.path.join(base_dir, folder)
        folder_info = {}
        total_rows = 0  # ⭐ 新增：统计总数

        if not os.path.exists(folder_path):
            report[folder] = {}
            report[folder]["_total"] = 0  # ⭐ 即使不存在也放一个总数
            continue

        # ----------------------------------------------------------------------
        # 特殊处理：01meta_data → 子目录，每个子目录内有一个 CSV
        # ----------------------------------------------------------------------
        if folder == "01meta_data":
            subdirs = [
                d for d in os.listdir(folder_path)
                if os.path.isdir(os.path.join(folder_path, d))
            ]

            for subdir in subdirs:
                subdir_path = os.path.join(folder_path, subdir)
                csv_files = [f for f in os.listdir(subdir_path) if f.endswith(".csv")]

                if not csv_files:
                    folder_info[subdir] = None
                    continue

                csv_file = csv_files[0]
                file_path = os.path.join(subdir_path, csv_file)

                try:
                    row_count = sum(1 for _ in open(file_path, "r", encoding="utf-8"))
                except UnicodeDecodeError:
                    row_count = sum(1 for _ in open(file_path, "r", encoding="gb18030"))

                folder_info[subdir] = row_count
                total_rows += row_count  # ⭐ 累加

            folder_info["_total"] = total_rows  # ⭐ 写入总数
            report[folder] = folder_info
            continue

        # ----------------------------------------------------------------------
        # 其他目录：直接扫描 CSV 文件
        # ----------------------------------------------------------------------
        csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

        for csv_file in csv_files:
            file_path = os.path.join(folder_path, csv_file)

            try:
                row_count = sum(1 for _ in open(file_path, "r", encoding="utf-8"))
            except UnicodeDecodeError:
                row_count = sum(1 for _ in open(file_path, "r", encoding="gb18030"))

            folder_info[csv_file] = row_count
            total_rows += row_count  # ⭐ 累加

        folder_info["_total"] = total_rows  # ⭐ 写入总数
        report[folder] = folder_info

    # ----------------------------------------------------------------------
    # 输出 JSON 文件
    # ----------------------------------------------------------------------
    output_path = os.path.join(base_dir, output_json)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return output_path


if __name__ == "__main__":
    count_csv_rows_to_json()

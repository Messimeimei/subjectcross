# -*- coding: utf-8 -*-
# Created by Messimeimei
# Updated with RBF-SVM 3-class classifier (2025/11/26)
"""
五维融合 → RBF-SVM → 3 分类任务
---------------------------------------
分类定义：
    0 = 无关学科（neither）
    1 = 交叉学科（cross）
    2 = 主学科（main）

输入特征：
    五维融合向量：incites, title_abs, author_aff, openalex, refs

步骤：
 1. 合并 test_data.csv + predicted_result.csv → 5dims_dataset.csv
 2. 解析五维字段 → paper_data
 3. 计算全局 min/max → 归一化
 4. 构建三分类训练集
 5. RBF-SVM 训练多分类模型
 6. 测试集预测 + Accuracy
"""

import os
import json
import ast
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle

# ================================================================
# Part 0 —— 合并数据，生成 5dims_dataset.csv
# ================================================================
def build_5dims_dataset(test_data_file: str,
                        predicted_data_file: str,
                        output_file: str):

    print("📥 读取 test_data.csv（真实标签）...")
    test_df = pd.read_csv(test_data_file, dtype=str).fillna("")

    print("📥 读取 predicted_result.csv（含五维字段）...")
    pred_df = pd.read_csv(predicted_data_file, dtype=str).fillna("")

    if "DOI" not in test_df.columns or "DOI" not in pred_df.columns:
        raise ValueError("test_data.csv 与 predicted_result.csv 必须包含 DOI 字段")

    print("🔄 按 DOI 合并数据 ...")
    merged = test_df.merge(pred_df, on="DOI", how="left", suffixes=("", "_pred"))

    required = [
        "DOI", "来源", "研究方向", "论文标题",
        "CR_摘要", "CR_作者和机构", "CR_参考文献DOI",
        "list_incites_direction", "list_title_abs",
        "list_author_aff_qwen", "list_openalex", "list_ref",
        "primary", "cross"
    ]

    for col in required:
        if col not in merged.columns:
            merged[col] = ""

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    merged[required].to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"🎉 5dims_dataset.csv 已生成，共 {len(merged)} 条")
    return merged


# ================================================================
# 工具函数：五维字段解析
# ================================================================
def safe_parse_list(s: str):
    if not isinstance(s, str) or s.strip() == "":
        return []
    try:
        return json.loads(s)
    except:
        try:
            return ast.literal_eval(s)
        except:
            return []


def extract_subject_code(field_name: str) -> str:
    if not isinstance(field_name, str) or field_name.strip() == "":
        return ""
    digits = "".join(c for c in field_name if c.isdigit())
    return digits[:4] if len(digits) >= 4 else ""


def clean_dim_items(raw_list):
    cleaned = []
    for item in raw_list:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            subject_code = extract_subject_code(str(item[0]))
            if subject_code == "":
                continue
            try:
                score = float(item[1])
            except:
                continue
            cleaned.append((subject_code, score))
    return cleaned


# ================================================================
# Part 2 —— 转换为 paper_data
# ================================================================
def convert_csv_to_paper_data(csv_path: str):
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    paper_data = []

    for _, row in df.iterrows():
        dims = {
            "incites": clean_dim_items(safe_parse_list(row["list_incites_direction"])),
            "title_abs": clean_dim_items(safe_parse_list(row["list_title_abs"])),
            "author_aff": clean_dim_items(safe_parse_list(row["list_author_aff_qwen"])),
            "openalex": clean_dim_items(safe_parse_list(row["list_openalex"])),
            "refs": clean_dim_items(safe_parse_list(row["list_ref"])),
        }

        main_label = extract_subject_code(row["primary"])
        cross_codes = [
            extract_subject_code(x)
            for x in row["cross"].split("；")
            if extract_subject_code(x) != ""
        ]

        paper_data.append({
            "paper_id": row["DOI"].strip(),
            "dims": dims,
            "label": {"main": main_label, "cross": cross_codes},
        })

    print(f"📦 已转换 paper_data，共 {len(paper_data)} 篇论文")
    return paper_data


# ================================================================
# Part 3 —— 全局 min / max
# ================================================================
def compute_global_min_max(paper_data):
    dim_names = ["incites", "title_abs", "author_aff", "openalex", "refs"]
    stats = {name: [] for name in dim_names}

    for paper in paper_data:
        dims = paper["dims"]
        for name in dim_names:
            for _, score in dims.get(name, []):
                stats[name].append(float(score))

    global_stats = {}
    print("\n📊 全局 min/max：")
    for name, values in stats.items():
        if not values:
            global_stats[name] = (0, 1)
            print(f" - {name} = EMPTY")
        else:
            mn, mx = min(values), max(values)
            global_stats[name] = (mn, mx)
            print(f" - {name}: min={mn:.4f}, max={mx:.4f}")

    return global_stats


def normalize_dim_with_stats(dim_list, min_v, max_v):
    if not dim_list:
        return {}
    if min_v == max_v:
        return {f: 0.0 for f, _ in dim_list}
    return {f: (float(s) - min_v) / (max_v - min_v) for f, s in dim_list}


# ================================================================
# Part 4 —— 3 分类训练集构建（核心）
# ================================================================
def build_dataset_3class(paper_data, global_stats):

    X_all, y_all, paper_index = [], [], []

    for paper in paper_data:
        dims = paper["dims"]

        inc = normalize_dim_with_stats(dims["incites"], *global_stats["incites"])
        tit = normalize_dim_with_stats(dims["title_abs"], *global_stats["title_abs"])
        aut = normalize_dim_with_stats(dims["author_aff"], *global_stats["author_aff"])
        ope = normalize_dim_with_stats(dims["openalex"], *global_stats["openalex"])
        ref = normalize_dim_with_stats(dims["refs"], *global_stats["refs"])

        fields = set(inc) | set(tit) | set(aut) | set(ope) | set(ref)
        if not fields:
            continue

        main = paper["label"]["main"]
        cross = set(paper["label"]["cross"])

        for f in fields:
            X_all.append([
                inc.get(f, 0.0),
                tit.get(f, 0.0),
                aut.get(f, 0.0),
                ope.get(f, 0.0),
                ref.get(f, 0.0),
            ])

            # ---------- 三分类标签 ----------
            if f == main:
                y = 2
            elif f in cross:
                y = 1
            else:
                y = 0
            # ---------------------------------

            y_all.append(y)
            paper_index.append((paper["paper_id"], f))

    print(f"📚 训练样本数：{len(X_all)}条")
    return np.array(X_all), np.array(y_all), paper_index


# ================================================================
# Part 5 —— RBF-SVM 三分类模型
# ================================================================
def train_rbf_svm_3class(X, y):
    print("\n🔥 使用 RBF-SVM 进行 3 分类训练 ...")

    model = SVC(
        kernel="rbf",
        probability=True,   # 必须打开才能 predict_proba
        C=2.0,
        gamma="scale",
        class_weight="balanced"   # ⭐ 核心修复

    )

    model.fit(X, y)

    print("🎉 RBF-SVM 训练完成")
    return model


# ================================================================
# Part 6 —— 测试集预测（三分类）
# ================================================================
def predict_3class(model, test_data, global_stats):
    results = []

    for paper in test_data:
        dims = paper["dims"]

        inc = normalize_dim_with_stats(dims["incites"], *global_stats["incites"])
        tit = normalize_dim_with_stats(dims["title_abs"], *global_stats["title_abs"])
        aut = normalize_dim_with_stats(dims["author_aff"], *global_stats["author_aff"])
        ope = normalize_dim_with_stats(dims["openalex"], *global_stats["openalex"])
        ref = normalize_dim_with_stats(dims["refs"], *global_stats["refs"])

        fields = set(inc) | set(tit) | set(aut) | set(ope) | set(ref)
        if not fields:
            continue

        true_main = paper["label"]["main"]
        true_cross = set(paper["label"]["cross"])

        for f in fields:
            x = np.array([[inc.get(f, 0), tit.get(f, 0),
                           aut.get(f, 0), ope.get(f, 0), ref.get(f, 0)]])
            prob = model.predict_proba(x)[0]
            pred = int(np.argmax(prob))

            real = 2 if f == true_main else (1 if f in true_cross else 0)

            results.append({
                "paper_id": paper["paper_id"],
                "field": f,
                "pred": pred,
                "real": real,
                "prob_0": prob[0],
                "prob_1": prob[1],
                "prob_2": prob[2],
            })

    return results


# ================================================================
# Part 7 —— 三分类准确率
# ================================================================
def compute_3class_accuracy(results):
    y_true = [r["real"] for r in results]
    y_pred = [r["pred"] for r in results]
    acc = accuracy_score(y_true, y_pred)
    print(f"\n🎯 3-class Accuracy = {acc:.4f}")
    print("\n📋 分类报告：")
    print(classification_report(y_true, y_pred, digits=4))
    return acc


# ================================================================
# 主入口
# ================================================================
if __name__ == "__main__":

    ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = ROOT / "data"

    test_data_file = DATA_DIR / "test_data.csv"
    pred_data_file = DATA_DIR / "predicted_result.csv"
    output_file = DATA_DIR / "5dims_dataset.csv"

    print("\n====== Step A：生成 5dims_dataset.csv ======")
    build_5dims_dataset(str(test_data_file), str(pred_data_file), str(output_file))

    print("\n====== Step B：转换为 paper_data ======")
    paper_data = convert_csv_to_paper_data(str(output_file))

    # 随机划分（可调）
    idx = np.arange(len(paper_data))
    np.random.seed(42)
    np.random.shuffle(idx)

    train_idx = idx[:30]
    test_idx = idx[30:]

    train_data = [paper_data[i] for i in train_idx]
    test_data = [paper_data[i] for i in test_idx]

    print(f"\n📌 Train: {len(train_data)} 篇, Test: {len(test_data)} 篇")

    print("\n====== Step C：计算全局 min/max ======")
    global_stats = compute_global_min_max(train_data)

    print("\n====== Step D：构建三分类训练集 ======")
    X, y, paper_index = build_dataset_3class(train_data, global_stats)

    print("\n====== Step E：训练 RBF-SVM ======")
    model = train_rbf_svm_3class(X, y)

    print("\n====== Step F：预测测试集 ======")
    results = predict_3class(model, test_data, global_stats)

    print("\n====== Step G：三分类准确率 ======")
    compute_3class_accuracy(results)

    print("\n🎉 任务完成！")

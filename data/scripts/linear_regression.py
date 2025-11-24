# Created by Messimeimei
# Updated with Global-Dim Normalization + Auto-Save LR Model (2025/11/21)

"""
跨学科判定五维融合训练脚本（全局归一化 + 自动模型保存版）
============================================================
本脚本用于从五个信息来源（五维度）融合判断论文主学科/交叉学科。

主要步骤：
1. 自动生成/更新五维融合训练文件 5dims_dataset.csv
2. 解析五维字段数据（五维得分）
3. 提取学科代码（统一为4位数字，如 1205 / 0812）
4. 将 CSV 转换为统一的 paper_data 结构
5. 基于“全局维度归一化”构造训练特征 X 与标签 y
6. 训练逻辑回归 LogisticRegression（五维融合）
7. 网格搜索最佳 threshold n 与 top-k
8. 自动保存模型（model.pkl, global_stats.json, best_params.json）
============================================================
"""

import os
import json
import ast
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import pickle


# ================================================================
# Part 0 —— 自动生成 5dims_dataset.csv
# ================================================================
def build_5dims_dataset(
    test_data_file: str,
    predicted_data_file: str,
    output_file: str = "../../data/5dims_dataset.csv"
):
    print("📥 读取 test_data.csv（真实标签）...")
    test_df = pd.read_csv(test_data_file, dtype=str).fillna("")

    print("📥 读取 predicted_result.csv（含五维字段）...")
    pred_df = pd.read_csv(predicted_data_file, dtype=str).fillna("")

    if "DOI" not in test_df.columns or "DOI" not in pred_df.columns:
        raise ValueError("test_data.csv 与 predicted_result.csv 必须包含 DOI 字段")

    print("🔄 按 DOI 合并数据 ...")
    merged = test_df.merge(pred_df, on="DOI", how="left", suffixes=("", "_pred"))

    required = [
        "DOI", "来源", "研究方向", "论文标题", "CR_摘要",
        "CR_作者和机构", "CR_参考文献DOI",
        "list_incites_direction", "list_title_abs",
        "list_author_aff_qwen", "list_openalex", "list_ref",
        "primary", "cross"
    ]
    for col in required:
        if col not in merged.columns:
            merged[col] = ""

    new_df = merged[required].copy()

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    new_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"🎉 5dims_dataset.csv 已生成/更新，共 {len(new_df)} 条")
    return new_df


# ================================================================
# Part 1 —— 五维字段解析
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
# Part 2 —— 5dims_dataset → paper_data
# ================================================================
def convert_csv_to_paper_data(csv_path: str):
    print(f"\n📥 加载 5dims_dataset.csv: {csv_path}")
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

    print(f"📦 转换完成，共 {len(paper_data)} 篇论文")
    return paper_data


# ================================================================
# Part 3 —— 全局 min/max
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

    print("\n📊 全局 min/max 统计：")
    for name, values in stats.items():
        if not values:
            print(f"  - {name}: EMPTY")
            global_stats[name] = (0, 1)
            continue
        mn, mx = min(values), max(values)
        print(f"  - {name}: min={mn:.4f}, max={mx:.4f}")
        global_stats[name] = (mn, mx)

    return global_stats


# ================================================================
# Part 4 —— 单维归一化
# ================================================================
def normalize_dim_with_stats(dim_list, min_v, max_v):
    if not dim_list:
        return {}
    if min_v == max_v:
        return {f: 0.0 for f, _ in dim_list}
    return {f: (float(s) - min_v) / (max_v - min_v) for f, s in dim_list}


# ================================================================
# Part 5 —— 构造训练集
# ================================================================
def build_dataset(paper_data, global_stats):
    X_all, y_all, paper_index = [], [], []

    for paper in paper_data:
        dims = paper["dims"]

        inc = normalize_dim_with_stats(dims["incites"], *global_stats["incites"])
        tit = normalize_dim_with_stats(dims["title_abs"], *global_stats["title_abs"])
        aut = normalize_dim_with_stats(dims["author_aff"], *global_stats["author_aff"])
        ope = normalize_dim_with_stats(dims["openalex"], *global_stats["openalex"])
        ref = normalize_dim_with_stats(dims["refs"], *global_stats["refs"])

        fields = set(inc) | set(tit) | set(aut) | set(ope) | set(ref)

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
            y_all.append(1 if (f == main or f in cross) else 0)
            paper_index.append((paper["paper_id"], f))

    return np.array(X_all), np.array(y_all), paper_index


# ================================================================
# Part 6 —— 训练逻辑回归
# ================================================================
def train_logistic(X, y):
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        max_iter=2000
    )
    model.fit(X, y)
    return model


# ================================================================
# Part 7 —— 网格搜索 n/k
# ================================================================
def search_threshold_k(model, paper_data, global_stats, thresholds=None, ks=None):
    if thresholds is None:
        thresholds = np.arange(0.2, 0.85, 0.05)
    if ks is None:
        ks = [1, 2, 3]

    best_f1, best_n, best_k = 0, 0.5, 1

    for n in thresholds:
        for k in ks:
            f1_list = []

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

                true_set = {paper["label"]["main"], *paper["label"]["cross"]}

                preds = []
                for f in fields:
                    x = np.array([[inc.get(f, 0), tit.get(f, 0),
                                   aut.get(f, 0), ope.get(f, 0), ref.get(f, 0)]])
                    prob = model.predict_proba(x)[0][1]
                    preds.append((f, prob))

                filtered = [(f, p) for f, p in preds if p > n]
                if not filtered:
                    continue

                filtered.sort(key=lambda x: x[1], reverse=True)
                pred_main = filtered[0][0]
                pred_cross = [f for f, _ in filtered[1:k+1]]
                pred_set = {pred_main, *pred_cross}

                y_true = [1 if f in true_set else 0 for f in fields]
                y_pred = [1 if f in pred_set else 0 for f in fields]
                f1_list.append(f1_score(y_true, y_pred))

            if f1_list:
                avg_f1 = float(np.mean(f1_list))
                if avg_f1 > best_f1:
                    best_f1, best_n, best_k = avg_f1, float(n), int(k)

    return best_n, best_k, best_f1


# ================================================================
# Part 8 —— 随机 30 训练 + 20 测试
# ================================================================
def split_train_test(paper_data, train_size=30, random_seed=42):
    """
    随机抽样：
    - 30 个论文训练
    - 剩余全部做预测
    """
    np.random.seed(random_seed)
    idx = np.arange(len(paper_data))
    np.random.shuffle(idx)

    train_idx = idx[:train_size]
    test_idx = idx[train_size:]

    train_data = [paper_data[i] for i in train_idx]
    test_data = [paper_data[i] for i in test_idx]

    print(f"\n📌 随机划分：训练 {len(train_data)} 篇，测试 {len(test_data)} 篇")
    return train_data, test_data


def train_pipeline_subset(train_data):
    print("\n=== Step 1：计算全局维度 min/max（基于训练集） ===")
    global_stats = compute_global_min_max(train_data)

    print("\n=== Step 2：构造训练数据 X/y ===")
    X, y, _ = build_dataset(train_data, global_stats)
    print(f"训练样本数: {len(X)}, 正样本比例: {y.mean():.4f}")

    print("\n=== Step 3：训练 LogisticRegression ===")
    model = train_logistic(X, y)
    print("五维 coef_:", model.coef_[0])
    print("偏置 intercept_:", model.intercept_[0])

    return model, global_stats


# ================================================================
# Part 9 —— 使用训练集 LR 模型预测测试集主学科
# ================================================================
def predict_main_subject(model, test_data, global_stats):
    results = []

    print("\n=== Step 4：对测试集进行预测（仅主学科） ===")

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

        preds = []
        for f in fields:
            x = np.array([[inc.get(f, 0), tit.get(f, 0),
                           aut.get(f, 0), ope.get(f, 0), ref.get(f, 0)]])
            prob = model.predict_proba(x)[0][1]
            preds.append((f, prob))

        preds.sort(key=lambda x: x[1], reverse=True)
        pred_main = preds[0][0]  # 取概率最大的

        results.append({
            "paper_id": paper["paper_id"],
            "real_main": paper["label"]["main"],      # 真实主学科
            "pred_main": pred_main,                   # 预测主学科
            "correct": pred_main == paper["label"]["main"],
            "top_prob": preds[0][1]
        })

    return results


# ================================================================
# 函数：计算主学科预测准确率
# ================================================================
def compute_accuracy(results):
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    acc = correct / total if total > 0 else 0
    print(f"\n🎯 主学科预测准确率：{acc:.4f}  ({correct}/{total})")
    return acc


# ================================================================
# 主入口（自动路径 + 30 训练 / 20 预测 + Accuracy）
# ================================================================
if __name__ == "__main__":

    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = ROOT / "data"

    test_data_file = DATA_DIR / "test_data.csv"
    predicted_data_file = DATA_DIR / "predicted_result.csv"
    output_file = DATA_DIR / "5dims_dataset.csv"

    print("\n====== Step A：生成 / 更新 5dims_dataset.csv ======")
    build_5dims_dataset(str(test_data_file), str(predicted_data_file), str(output_file))

    print("\n====== Step B：转换为 paper_data ======")
    paper_data = convert_csv_to_paper_data(str(output_file))

    # -----------------------------
    # 🎯 随机划分 30 / 剩余 20
    # -----------------------------
    train_data, test_data = split_train_test(paper_data, train_size=30)

    # -----------------------------
    # 🎯 对 30 个训练
    # -----------------------------
    model, global_stats = train_pipeline_subset(train_data)

    # -----------------------------
    # 🎯 对剩下 test_data 做预测
    # -----------------------------
    results = predict_main_subject(model, test_data, global_stats)

    print("\n====== 逐条预测结果 ======")
    for r in results:
        print(f"{r['paper_id']} | 真={r['real_main']} | 预测={r['pred_main']} | "
              f"prob={r['top_prob']:.4f} | correct={r['correct']}")

    # -----------------------------
    # 🎯 计算主学科 Accuracy
    # -----------------------------
    compute_accuracy(results)

    print("\n🎉 完成！")

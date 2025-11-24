# -*- coding: utf-8 -*-
# Created by Messimeimei
# Rebuilt by ChatGPT — Field-level 3-class GBDT (2025/12)

import os
import json
import ast
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, log_loss

import matplotlib.pyplot as plt


# ==========================================================
# 工具函数：安全解析 list/json 字段
# ==========================================================
def safe_parse_list(s):
    if not isinstance(s, str) or s.strip() == "":
        return []
    try:
        return json.loads(s)
    except:
        try:
            return ast.literal_eval(s)
        except:
            return []


# 从 "1205 信息资源管理" / "1205" / "['1205',0.7]" 中抽取 4 位码
def extract_subject_code(field_name):
    if field_name is None:
        return ""
    s = str(field_name)
    digits = "".join([c for c in s if c.isdigit()])
    return digits[:4] if len(digits) >= 4 else ""


# 解析五维度字段：[(学科名, 分数)] → [(code, score)]
def clean_dim_items(raw):
    cleaned = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        code = extract_subject_code(item[0])
        if code == "":
            continue
        try:
            score = float(item[1])
        except:
            continue
        cleaned.append((code, score))
    return cleaned


# ==========================================================
# Part A —— 构建 5dims_dataset.csv
# ==========================================================
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


# ==========================================================
# Part B —— 转成 paper_data
# ==========================================================
def convert_csv_to_paper_data(csv_path):
    print(f"\n📥 加载 5dims_dataset.csv: {csv_path}")
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    papers = []

    for _, row in df.iterrows():
        dims = {
            "incites": clean_dim_items(safe_parse_list(row["list_incites_direction"])),
            "title_abs": clean_dim_items(safe_parse_list(row["list_title_abs"])),
            "author_aff": clean_dim_items(safe_parse_list(row["list_author_aff_qwen"])),
            "openalex": clean_dim_items(safe_parse_list(row["list_openalex"])),
            "refs": clean_dim_items(safe_parse_list(row["list_ref"])),
        }

        main = [
            extract_subject_code(x)
            for x in row["primary"].replace("；", ";").split(";")
            if extract_subject_code(x) != ""
        ]
        cross = [
            extract_subject_code(x)
            for x in row["cross"].replace("；", ";").split(";")
            if extract_subject_code(x) != ""
        ]

        papers.append({
            "paper_id": row["DOI"],
            "dims": dims,
            "label": {
                "main": main,
                "cross": cross,
            }
        })

    print(f"📦 转换完成，共 {len(papers)} 篇论文")
    return papers


# ==========================================================
# Part C —— 计算五维度全局 min/max
# ==========================================================
def compute_global_min_max(paper_data):
    dim_names = ["incites", "title_abs", "author_aff", "openalex", "refs"]
    stats = {d: [] for d in dim_names}

    for p in paper_data:
        for d in dim_names:
            stats[d].extend([float(v) for _, v in p["dims"][d]])

    global_stats = {}
    print("\n📊 全局 min/max：")
    for d, vals in stats.items():
        if len(vals) == 0:
            global_stats[d] = (0.0, 1.0)
        else:
            global_stats[d] = (min(vals), max(vals))
        print(f" - {d}: {global_stats[d]}")
    return global_stats


def normalize_dim_with_stats(dim_list, mn, mx):
    if not dim_list:
        return {}
    if mn == mx:
        return {f: 0.0 for f, _ in dim_list}
    mn = 0
    return {f: (float(s) - mn) / (mx - mn) for f, s in dim_list}


# ==========================================================
# Part D —— 构建学科级别 3 分类数据（最核心部分）
# ==========================================================
def build_dataset_3class(paper_data, stats, stage="train"):

    X_all, y_all, samples = [], [], []

    for p in paper_data:

        # 五维归一化
        inc = normalize_dim_with_stats(p["dims"]["incites"], *stats["incites"])
        tit = normalize_dim_with_stats(p["dims"]["title_abs"], *stats["title_abs"])
        aut = normalize_dim_with_stats(p["dims"]["author_aff"], *stats["author_aff"])
        ope = normalize_dim_with_stats(p["dims"]["openalex"], *stats["openalex"])
        ref = normalize_dim_with_stats(p["dims"]["refs"], *stats["refs"])


        # 五维度中出现的所有学科
        fields = set(inc) | set(tit) | set(aut) | set(ope) | set(ref)

        # 强制加入主学科与交叉学科（避免缺失）
        main = p["label"]["main"]
        cross = set(p["label"]["cross"])


        # 构建学科样本
        for f in fields:
            feat = [
                inc.get(f, 0.0),
                tit.get(f, 0.0),
                aut.get(f, 0.0),
                ope.get(f, 0.0),
                ref.get(f, 0.0),
            ]

            if f in main:
                y = 2
            elif f in cross:
                y = 1
            else:
                y = 0

            X_all.append(feat)
            y_all.append(y)
            samples.append({
                "paper_id": p["paper_id"],
                "field": f,
                "feat": feat,
                "label": y,
            })

    print(f"\n📘 [{stage}] 三分类样本数：{len(X_all)}")

    if stage == "train":
        print("\n====== 🔍【训练样本示例（前 10 条）】======")
        for s in samples[:10]:
            print(f"{s['paper_id']} | field={s['field']} | y={s['label']} | feat={s['feat']}")

        y_np = np.array(y_all)
        print("\n====== 🔢【训练集类别分布】======")
        print(f"0（无关）: {np.sum(y_np==0)}")
        print(f"1（交叉）: {np.sum(y_np==1)}")
        print(f"2（主学科）: {np.sum(y_np==2)}")
    
    print(f"训练样本：{X_all[:10], y_all[:10]}")

    return np.array(X_all), np.array(y_all), samples


# ==========================================================
# Part E —— 训练 GBDT 三分类
# ==========================================================
def train_gbdt(X, y):

    # print("\n🔥 使用 RBF-SVM 进行 3 分类训练 ...")

    # model = SVC(
    #     kernel="rbf",
    #     probability=True,   # 必须打开才能 predict_proba
    #     C=2.0,
    #     gamma="scale",
    #     class_weight="balanced"   # ⭐ 核心修复

    # )

    # model.fit(X, y)

    # print("🎉 RBF-SVM 训练完成")
    # return model

    print("\n🔥 使用 GBDT 训练三分类模型 ...")

    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,         # 使用部分样本（更稳定）

    )
    model.fit(X, y)

    print("🎉 GBDT 训练完成")

    print("\n====== 📉【训练 Loss】======")
    for i, y_pred in enumerate(model.staged_predict_proba(X)):
        if i % 50 == 0:
            loss = log_loss(y, y_pred, labels=[0,1,2])
            print(f"Iter {i:3d} | loss={loss:.4f}")

    return model


# ==========================================================
# Part F —— 学科级别评估（你要的结果）
# ==========================================================
def evaluate_field_level_3class(y_true, y_pred):
    print("\n====== 📊【字段级三分类报告】======")
    print(classification_report(y_true, y_pred, digits=4))

    print("\n====== 📊【混淆矩阵】======")
    print(confusion_matrix(y_true, y_pred))


# ==========================================================
# 主入口
# ==========================================================
if __name__ == "__main__":

    ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = ROOT / "data"

    test_file = DATA_DIR / "test_data.csv"
    pred_file = DATA_DIR / "predicted_result.csv"
    dims_file = DATA_DIR / "5dims_dataset.csv"

    # Step 1: 构建 5dims_dataset
    build_5dims_dataset(test_file, pred_file, dims_file)

    # Step 2: 解析成 paper_data
    papers = convert_csv_to_paper_data(dims_file)

    # Step 3: 论文级划分（例如 40 / 10）
    idx = np.arange(len(papers))
    np.random.shuffle(idx)
    train_papers = [papers[i] for i in idx[:45]]
    test_papers  = [papers[i] for i in idx[45:]]

    # Step 4: 全局 min-max（基于训练论文）
    stats = compute_global_min_max(train_papers)

    # Step 5: 构建学科级训练集
    X_train, y_train, train_samples = build_dataset_3class(train_papers, stats, stage="train")

    # Step 6: 训练模型
    model = train_gbdt(X_train, y_train)

    # Step 7: 构建学科级测试集
    X_test, y_test, test_samples = build_dataset_3class(test_papers, stats, stage="test")

    # Step 8: 测试评估（你要的结果）
    y_pred = model.predict(X_test)
    evaluate_field_level_3class(y_test, y_pred)

    print("\n🎉 完成！")

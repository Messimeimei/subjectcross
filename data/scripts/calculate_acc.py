import json
import os

import pandas as pd
from tqdm import tqdm
from utils.subject_calculator import SubjectCalculator
import matplotlib.pyplot as plt
import numpy as np


def align_predicted_with_test(test_data_file, predicted_data_file):
    """根据 DOI 匹配 test_data 和 predicted_result，使两者行对应"""
    test_df = pd.read_csv(test_data_file, dtype=str).fillna("")
    pred_df = pd.read_csv(predicted_data_file, dtype=str).fillna("")

    # 确保存在 DOI 列
    if "DOI" not in test_df.columns or "DOI" not in pred_df.columns:
        raise ValueError("两个文件必须都包含 'DOI' 列")

    # 过滤 predicted_result，只保留出现在 test_data 中的 DOI
    test_dois = set(test_df["DOI"].unique())
    aligned_pred_df = pred_df[pred_df["DOI"].isin(test_dois)].copy()

    # 重新排序，使 predicted 文件的顺序与 test_data 一致
    aligned_pred_df = aligned_pred_df.set_index("DOI").reindex(test_df["DOI"]).reset_index()

    # 覆盖保存 predicted_result.csv
    aligned_pred_df.to_csv(predicted_data_file, index=False, encoding="utf-8-sig")

    print(f"✅ 已对齐 predicted_result.csv，共保留 {len(aligned_pred_df)} 条记录")
    return test_df, aligned_pred_df


def mark_wrong_predictions(test_data, predicted_data, output_dir="data"):
    """
    标记预测错误的论文：
    - 主学科预测错误 → main_wrong
    - 交叉学科F1=0 → cross_wrong
    - 两者同时出现时用中文分号连接
    """
    test_data_with_mark = test_data.copy()
    test_data_with_mark['predict_wrong'] = ''

    columns_to_keep = ['DOI', '来源', '研究方向', 'primary', 'cross', 'detail']
    existing_columns = [col for col in columns_to_keep if col in predicted_data.columns]

    predicted_data_with_mark = predicted_data[existing_columns].copy()
    predicted_data_with_mark['predict_wrong'] = ''

    wrong_count_main = 0
    wrong_count_cross = 0

    for _, test_row in test_data.iterrows():
        doi = test_row['DOI']
        test_primary = [item.strip()[:4] for item in test_row['primary'].split('；') if item.strip()]
        test_cross = [item.strip()[:4] for item in test_row['cross'].split('；') if item.strip()]
        predicted_row = predicted_data[predicted_data['DOI'] == doi]
        if predicted_row.empty:
            continue

        predicted_row = predicted_row.iloc[0]
        predicted_primary = [item.strip()[:4] for item in predicted_row['primary'].split(',') if item.strip()]
        predicted_cross = [item.strip()[:4] for item in predicted_row['cross'].split(',') if item.strip()]

        # --- 主学科正确性 ---
        primary_score = 1 if any(item in test_primary for item in predicted_primary) else 0

        # --- 交叉学科新的准确率 / 召回率 / F1 ---
        test_all_labels = set(test_primary + test_cross)
        predicted_cross_set = set(predicted_cross)

        # 处理测试交叉学科为空的情况
        if not test_cross and not predicted_cross_set:
            # 如果测试交叉学科为空，预测的也为空，则准确率、召回率、F1都为满分
            cross_acc = 1.0
            cross_rec = 1.0
            f1_score = 1.0
        elif not test_cross and predicted_cross_set:
            # 如果测试交叉学科为空，但预测的不为空，则准确率为0，召回率为1，F1为0
            cross_acc = 0.0
            cross_rec = 1.0
            f1_score = 0.0
        elif test_cross and not predicted_cross_set:
            # 如果测试交叉学科不为空，但预测的为空，则准确率为1，召回率为0，F1为0
            cross_acc = 1.0
            cross_rec = 0.0
            f1_score = 0.0
        else:
            # 正常情况：两者都不为空
            correct_cross = len(predicted_cross_set & test_all_labels)
            cross_acc = correct_cross / len(predicted_cross_set) if predicted_cross_set else 0
            cross_rec = correct_cross / len(test_all_labels) if test_all_labels else 0
            f1_score = (2 * cross_acc * cross_rec) / (cross_acc + cross_rec) if cross_acc + cross_rec else 0

        # --- 错误标记 ---
        mark_labels = []
        if primary_score == 0:
            mark_labels.append("main_wrong")
            wrong_count_main += 1
        if f1_score == 0:
            mark_labels.append("cross_wrong")
            wrong_count_cross += 1

        mark_str = '；'.join(mark_labels) if mark_labels else ''
        if mark_str:
            test_data_with_mark.loc[test_data_with_mark['DOI'] == doi, 'predict_wrong'] = mark_str
            predicted_data_with_mark.loc[predicted_data_with_mark['DOI'] == doi, 'predict_wrong'] = mark_str

    # --- 保存结果 ---
    test_output_file = os.path.join(output_dir, "test_data_marked.csv")
    predicted_output_file = os.path.join(output_dir, "predicted_result_marked.csv")

    test_data_with_mark.to_csv(test_output_file, index=False, encoding="utf-8-sig")
    predicted_data_with_mark.to_csv(predicted_output_file, index=False, encoding="utf-8-sig")

    print(f"💾 已生成标记文件: test_data_marked.csv 与 predicted_result_marked.csv")
    print(f"📊 主学科预测错误(main_wrong): {wrong_count_main}")
    print(f"📊 交叉学科F1=0 (cross_wrong): {wrong_count_cross}")


def refrank_with_llm_and_stub_result(input_dir="data/04input_data", output_dir="data/05output_data", file_path=None):
    """阶段四：直接计算学科结果（可指定单个文件）"""
    os.makedirs(output_dir, exist_ok=True)
    files = [os.path.basename(file_path)] if file_path else sorted(
        [f for f in os.listdir(input_dir) if f.endswith(".csv")]
    )
    if file_path:
        input_dir = os.path.dirname(file_path)

    calc = SubjectCalculator(debug=True, strategy="lr", topk_cross=3)
    out_csv = os.path.join(output_dir, "predicted_result.csv")
    results_json = []

    for fname in files:
        in_csv = os.path.join(input_dir, fname)
        print(f"➡️ 正在处理文件: {fname}")

        df = pd.read_csv(in_csv, dtype=str).fillna("")
        for _, row in tqdm(df.iterrows(), total=len(df), desc="  ⚙️ 计算", ncols=90, leave=False):
            try:
                res = calc.calc(row)
                results_json.append(json.dumps(res, ensure_ascii=False))
            except Exception:
                results_json.append("{}")

        df["result"] = results_json
        df["primary"] = df["result"].apply(lambda x: json.loads(x).get("primary"))
        df["cross"] = df["result"].apply(lambda x: ",".join(json.loads(x).get("cross", [])))
        df["detail"] = df["result"].apply(lambda x: json.dumps(json.loads(x).get("detail", {}), ensure_ascii=False))
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    return df


def calculate_accuracy(test_data_file, predicted_data_file, output_dir="data"):
    """计算准确率、召回率和F1分数，并打印每篇论文的详细计算结果"""
    test_data = pd.read_csv(test_data_file, dtype=str).fillna("")
    predicted_data = pd.read_csv(predicted_data_file, dtype=str).fillna("")

    primary_correct_count = 0
    total_primary_count = len(test_data)

    primary_correct_list, cross_accuracy_list, cross_recall_list, f1_score_list = [], [], [], []

    print("\n📊 ========== 每篇论文的计算结果 ==========")
    for i, test_row in enumerate(test_data.itertuples(), 1):
        test_primary = [item.strip()[:4] for item in getattr(test_row, 'primary').split('；') if item.strip()]
        test_cross = [item.strip()[:4] for item in getattr(test_row, 'cross').split('；') if item.strip()]
        predicted_row = predicted_data[predicted_data['DOI'] == getattr(test_row, 'DOI')]
        if predicted_row.empty:
            continue
        predicted_row = predicted_row.iloc[0]
        predicted_primary = [item.strip()[:4] for item in predicted_row['primary'].split(',') if item.strip()]
        predicted_cross = [item.strip()[:4] for item in predicted_row['cross'].split(',') if item.strip()]

        # === 主学科准确性 ===
        primary_score = 1 if any(item in test_primary for item in predicted_primary) else 0
        primary_correct_count += primary_score

        # === 修改后的交叉学科准确率 / 召回率计算 ===
        test_all_labels = set(test_primary + test_cross)
        predicted_cross_set = set(predicted_cross)

        # 处理测试交叉学科为空的情况
        if not test_cross and not predicted_cross_set:
            # 如果测试交叉学科为空，预测的也为空，则准确率、召回率、F1都为满分
            cross_accuracy = 1.0
            cross_recall = 1.0
            f1_score = 1.0
        elif not test_cross and predicted_cross_set:
            # 如果测试交叉学科为空，但预测的不为空，则准确率为0，召回率为1，F1为0
            cross_accuracy = 0.0
            cross_recall = 1.0
            f1_score = 0.0
        elif test_cross and not predicted_cross_set:
            # 如果测试交叉学科不为空，但预测的为空，则准确率为1，召回率为0，F1为0
            cross_accuracy = 1.0
            cross_recall = 0.0
            f1_score = 0.0
        else:
            # 正常情况：两者都不为空
            correct_cross = len(predicted_cross_set & test_all_labels)
            cross_accuracy = correct_cross / len(predicted_cross_set) if predicted_cross_set else 0
            cross_recall = correct_cross / len(test_all_labels) if test_all_labels else 0
            f1_score = (2 * cross_accuracy * cross_recall) / (
                        cross_accuracy + cross_recall) if cross_accuracy + cross_recall else 0

        primary_correct_list.append(primary_score)
        cross_accuracy_list.append(cross_accuracy)
        cross_recall_list.append(cross_recall)
        f1_score_list.append(f1_score)

        # === 打印每篇论文的详细结果 ===
        print(f"\n📄 [{i}] DOI: {getattr(test_row, 'DOI', '(无)')}")
        print(f"  🏷️ 测试主学科: {test_primary}")
        print(f"  🧠 预测主学科: {predicted_primary} {'✅' if primary_score else '❌'}")
        print(f"  🔗 测试交叉学科: {test_cross}")
        print(f"  🔍 预测交叉学科: {predicted_cross}")
        print(f"     ↳ 准确率={cross_accuracy:.3f}  召回率={cross_recall:.3f}  F1={f1_score:.3f}")

    # === 汇总统计 - 使用平均值 ===
    primary_accuracy = primary_correct_count / total_primary_count if total_primary_count else 0
    cross_accuracy = np.mean(cross_accuracy_list) if cross_accuracy_list else 0
    cross_recall = np.mean(cross_recall_list) if cross_recall_list else 0
    cross_f1 = np.mean(f1_score_list) if f1_score_list else 0

    print("\n📈 ======= 总体指标汇总 =======")
    print(f"主学科准确率: {primary_accuracy:.4f}")
    print(f"交叉学科准确率: {cross_accuracy:.4f}")
    print(f"交叉学科召回率: {cross_recall:.4f}")
    print(f"交叉学科F1分数: {cross_f1:.4f}")

    # 保存带标记文件
    mark_wrong_predictions(test_data, predicted_data, output_dir)
    plot_accuracy(primary_correct_list, cross_accuracy_list, cross_recall_list, f1_score_list)


def plot_accuracy(primary_correct_list, cross_accuracy_list, cross_recall_list, f1_score_list):
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    n, bins, patches = plt.hist(primary_correct_list, bins=[0, 1, 2], edgecolor="black", color="skyblue")
    plt.title("Primary Discipline Match")
    plt.xticks([0.5, 1.5], ["Not Matched", "Matched"])
    plt.xlabel("Match Status")
    plt.ylabel("Count")
    for patch in patches:
        plt.text(patch.get_x() + patch.get_width() / 2, patch.get_height() + 0.1, f'{int(patch.get_height())}',
                 ha='center')

    for i, (title, data, color) in enumerate(
            [("Cross Discipline Accuracy", cross_accuracy_list, "salmon"),
             ("Cross Discipline Recall", cross_recall_list, "lightgreen"),
             ("Cross Discipline F1 Score", f1_score_list, "orange")], start=2):
        plt.subplot(2, 2, i)
        bins = np.arange(0, 1.1, 0.1)
        n, bins, patches = plt.hist(data, bins=bins, edgecolor="black", color=color)
        plt.title(title)
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.xlabel(title.split()[-1])
        plt.ylabel("Count")
        for patch in patches:
            plt.text(patch.get_x() + patch.get_width() / 2, patch.get_height() + 0.1, f'{int(patch.get_height())}',
                     ha='center')

    plt.tight_layout()
    plt.show()


def main():
    input_data_dir = '../../data/04input_data'
    output_data_dir = '../../data'
    test_data_file = '../../data/test_data.csv'
    predicted_data_file = os.path.join(output_data_dir, "predicted_result.csv")

    print("开始执行预测结果生成...")
    refrank_with_llm_and_stub_result(input_dir=input_data_dir, output_dir=output_data_dir,
                                     file_path='../../data/04input_data/1205 Library and Information Science & Archive Management.csv')

    print("开始对齐 predicted_result.csv 与 test_data.csv ...")
    test_data, aligned_pred_data = align_predicted_with_test(test_data_file, predicted_data_file)

    print("开始计算准确率...")
    calculate_accuracy(test_data_file, predicted_data_file, output_dir=output_data_dir)


if __name__ == "__main__":
    main()
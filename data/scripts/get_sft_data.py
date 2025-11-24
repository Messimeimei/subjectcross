# -*- coding: utf-8 -*-
# Created by Messimeimei
# Modified by ChatGPT — conversation-style dataset (2025/12)
"""
本脚本提供完整训练数据生成流水线（conversation 格式）：

原始版本输出样本格式：
    {
        "instruction": "...",
        "input": "题名：...\n摘要：...",
        "output": {
            "primary": "...",
            "cross": [...]
        }
    }

改造后输出为对话格式：
    {
        "conversation_id": "xxx",
        "category": "主/交叉学科判定（117 一级学科，多标签）",
        "conversation": [
            {
                "human": "<instruction + input>",
                "assistant": "<output 的 JSON 文本>"
            }
        ]
    }

流水线步骤保持不变：
1. 读取 ../05output_data 下的每个学科 CSV
2. 从 CSV 中读取 primary / cross 字段
3. 转换为 JSONL（每学科独立，conversation 格式）
4. 每个学科独立按 0.6 / 0.2 / 0.2 划分 train/val/test
5. 汇总生成全局 train.jsonl / val.jsonl / test.jsonl
6. 自动绘制 4 张可视化图：
   - 数据集规模柱状图
   - 主学科分布图
   - 输入长度分布（字符数）
   - Token 长度分布
"""

import os
import json
import random
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
from transformers import AutoTokenizer

from pathlib import Path
from transformers import AutoTokenizer

# ================================
# 全局配置
# ================================

SPLIT_RATIO = (0.6, 0.2, 0.2)
SEED = 42

# 用当前脚本位置推断项目根目录，再定位本地模型
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]          # ~/pyprojects/subjectcross
TOKENIZER_PATH = PROJECT_ROOT / "models/base/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH), trust_remote_code=True)
MAX_CROSS = 3
INPUT_DIR = PROJECT_ROOT / "data/05output_data"
OUTPUT_DIR = PROJECT_ROOT / "data/06finetune_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# category 字段：统一写死
CATEGORY = "主/交叉学科判定（117 一级学科，多标签）"

# ================================
# Prompt 构造：新版（无分数）
# ================================
disciplines_df = pd.read_csv(PROJECT_ROOT / "data/zh_disciplines.csv", encoding="utf-8")
discipline_list = (
    disciplines_df.columns.tolist() +
    disciplines_df.iloc[:, 0].tolist()
)



# ================================
# 工具函数
# ================================
def safe_json_loads(text):
    try:
        return json.loads(text)
    except Exception:
        return None


def write_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_conversation_id(subject: str, idx: int, row: pd.Series) -> str:
    """
    生成 conversation_id：
    - 优先使用 DOI："<DOI>$No<idx>"
    - 没有 DOI 就用 "<subject>$No<idx>"
    """
    doi = str(row.get("DOI", "")).strip()
    if doi:
        return f"{doi}$No{idx}"
    return f"{subject}$No{idx}"


# ================================
# CSV → conversation 样本列表
# ================================
def convert_csv(csv_path, subject_name: str):
    """
    从单个学科的 CSV 构造 conversation 样本：
    {
        "conversation_id": "...",
        "category": CATEGORY,
        "conversation": [
            {
                "human": INSTRUCTION + "\\n\\n" + "题名：...\\n摘要：...",
                "assistant": "<output JSON 字符串>"
            }
        ]
    }
    """
    df = pd.read_csv(csv_path, encoding="utf-8").fillna("")
    samples = []

    for idx, (_, row) in enumerate(
        tqdm(df.iterrows(), total=len(df), desc=f"解析 {os.path.basename(csv_path)}")
    ):
        title = row.get("论文标题", "").strip()
        abstract = row.get("CR_摘要", "").strip()

        # --- 直接读取 CSV 中的 primary / cross 字段 ---
        primary = row.get("primary", "").strip()
        cross_raw = row.get("cross", "").strip()

        if not primary:
            # 没有主学科的样本跳过
            continue

        # cross 字段：兼容中英文逗号和分号
        cross = []
        if cross_raw:
            tmp = (
                cross_raw.replace("；", ",")
                .replace("，", ",")
                .split(",")
            )
            cross = [c.strip() for c in tmp if c.strip()]

        # ===== 构造原始 SFT 三元组（仅作为中间形态） =====
        # 1）human = 原来的 instruction + input
        human_text = (
            "你是一名论文学科的分类专家，擅长通过给定的论文标题和摘要，判断出论文的主学科和涉及的交叉学科。"
            "现在给你一篇论文的标题和摘要，请判断："
            " 1）论文的主学科（primary）"
            " 2）论文涉及的交叉学科（cross），最多 3 个，可为空\n"
            "【论文信息】\n"
            f"论文标题: {title}\n"
            f"论文摘要: {abstract}\n"
            "【输出格式要求】\n"
            "• 输出格式为 JSON，不能包含额外解释。\n"
            "• primary 必须且只能有 1 个，格式：'代码 学科名'\n"
            "• cross 是数组，每个元素为 代码 学科名（最多 3 个）\n"
            "【输出示例】\n"
            "{"
            "'primary': '0812 计算机科学与技术',"
            "'cross': ['0831 生物医学工程', '0702 物理学']"
            "}"
            )

        output_obj = {
            "primary": primary,
            "cross": cross
        }

        # 3）conversation_id & category
        conv_id = build_conversation_id(subject_name, idx, row)

        sample = {
            "conversation_id": conv_id,
            "category": CATEGORY,
            "conversation": [
                {
                    "human": human_text,
                    "assistant": json.dumps(output_obj, ensure_ascii=False) # 转成字符串
                }
            ]
        }
        samples.append(sample)

    return samples


# ================================
# 每学科拆分 train/val/test
# ================================
def split_by_ratio(samples):
    random.seed(SEED)
    random.shuffle(samples)

    n = len(samples)
    n_train = int(n * SPLIT_RATIO[0])
    n_val = int(n * SPLIT_RATIO[1])
    n_test = n - n_train - n_val

    train = samples[:n_train]
    val = samples[n_train:n_train + n_val]
    test = samples[n_train + n_val:]

    return train, val, test


# ================================
# 可视化工具（基于 conversation）
# ================================
def plot_dataset_statistics(train, val, test):
    # 设置全局英文字体
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.unicode_minus'] = False  # 避免负号显示异常

    # 图 1：比例
    plt.figure(figsize=(6, 5))
    plt.bar(
        ["train", "val", "test"],
        [len(train), len(val), len(test)],
        color=["#4e79a7", "#f28e2c", "#e15759"]
    )
    plt.title("Train / Validation / Test Sample Count")
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    plt.savefig("chart_dataset_split.png", dpi=160)
    plt.close()

    # 图 2：主学科分布（Top 20）
    def extract_primary(sample):
        try:
            assist = sample["conversation"][0]["assistant"]
            obj = safe_json_loads(assist)
            if isinstance(obj, dict):
                return obj.get("primary", "")
        except Exception:
            return ""
        return ""

    primary_list = [extract_primary(x) for x in train]
    primary_list = [p for p in primary_list if p]
    if primary_list:
        counter = Counter(primary_list).most_common(20)
        labels, values = zip(*counter)

        plt.figure(figsize=(10, 7))
        plt.barh(labels[::-1], values[::-1], color="#4e79a7")
        plt.title("Top 20 Primary Disciplines (Train Set)")
        plt.xlabel("Number of Samples")
        plt.tight_layout()
        plt.savefig("chart_primary_distribution.png", dpi=160)
        plt.close()

    # 图 3：输入文本长度（字符）——这里用 human（instruction+input）
    input_lens = [len(x["conversation"][0]["human"]) for x in train]

    plt.figure(figsize=(8, 5))
    plt.hist(input_lens, bins=50, color="#59a14f")
    plt.title("Distribution of Input Text Length (Characters)")
    plt.xlabel("Character Count")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("chart_input_length.png", dpi=160)
    plt.close()

    # 图 4：token 分布（对 human 编码）
    token_lens = [
        len(tokenizer.encode(x["conversation"][0]["human"]))
        for x in train
    ]

    plt.figure(figsize=(8, 5))
    plt.hist(token_lens, bins=50, color="#af7aa1")
    plt.title("Distribution of Input Token Length")
    plt.xlabel("Token Count")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("chart_input_token_length.png", dpi=160)
    plt.close()

    print("🎉 Visualization completed (conversation format, English fonts).")


# ================================
# 主流程
# ================================
if __name__ == "__main__":
    train_all, val_all, test_all = [], [], []

    for csv_file in os.listdir(INPUT_DIR):
        if not csv_file.endswith(".csv"):
            continue

        subject = csv_file.replace(".csv", "")
        csv_path = os.path.join(INPUT_DIR, csv_file)

        print(f"\n=== 处理学科：{subject} ===")
        samples = convert_csv(csv_path, subject)

        if not samples:
            print(f"⚠️ 学科 {subject} 无有效样本")
            continue

        train_s, val_s, test_s = split_by_ratio(samples)

        # 每学科独立文件（conversation 格式）
        write_jsonl(os.path.join(OUTPUT_DIR, f"{subject}_train.jsonl"), train_s)
        write_jsonl(os.path.join(OUTPUT_DIR, f"{subject}_val.jsonl"), val_s)
        write_jsonl(os.path.join(OUTPUT_DIR, f"{subject}_test.jsonl"), test_s)

        train_all.extend(train_s)
        val_all.extend(val_s)
        test_all.extend(test_s)

    # 全局合并（conversation 格式）
    write_jsonl(os.path.join(OUTPUT_DIR, "train.jsonl"), train_all)
    write_jsonl(os.path.join(OUTPUT_DIR, "val.jsonl"), val_all)
    write_jsonl(os.path.join(OUTPUT_DIR, "test.jsonl"), test_all)

    print("\n🎯 全部完成！（conversation 格式）")
    print(f"Train: {len(train_all)}   Val: {len(val_all)}   Test: {len(test_all)}")

    # 生成可视化图
    plot_dataset_statistics(train_all, val_all, test_all)

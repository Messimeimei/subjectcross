# Created by Messimeimei
# Updated by ChatGPT (2025/11)
# -*- coding: utf-8 -*-
"""
批量运行 getmeta / getref / getinput / getrank
支持：
  - 从 txt 中读取学科（含 |1 跳过）
  - 可以选择从哪个阶段开始，到哪个阶段结束
  - getmeta 读取目录；其余阶段读取 CSV 文件
"""

import argparse
from pathlib import Path
import sys
import main


# ======================================================
# 读取 txt 学科列表
# ======================================================

def load_subjects_from_txt(txt_path: str):
    path = Path(txt_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"❌ 学科列表文件不存在: {path}")

    subjects_to_run, subjects_skipped = [], []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [p.strip() for p in line.split("|")]
            subject = parts[0]
            skip_flag = len(parts) > 1 and parts[1] == "1"

            if skip_flag:
                subjects_skipped.append(subject)
            else:
                subjects_to_run.append(subject)

    print(f"📄 读取 {len(subjects_to_run)+len(subjects_skipped)} 行 (运行 {len(subjects_to_run)} | 跳过 {len(subjects_skipped)})")
    return subjects_to_run, subjects_skipped


# ======================================================
# 通用阶段执行逻辑（自动判断目录/文件）
# ======================================================

def run_stage(subjects, stage, root_dir):
    """
    stage: getmeta / getref / getinput / getrank
    root_dir: .../01meta_data or .../02crossref_data ...
    """

    print("\n" + "=" * 80)
    print(f"🚀 执行阶段：{stage}  |  基础路径: {root_dir}")
    print("=" * 80)

    root = Path(root_dir).resolve()
    ok, skip = 0, 0

    for subject in subjects:
        # --------------------------------------------------
        # getmeta → 目录路径
        # 其他阶段 → 文件路径
        # --------------------------------------------------
        if stage == "getmeta":
            target = root / subject  # 目录
            is_valid = target.exists() and target.is_dir()
            call_kwargs = dict(mode=stage, dir_path=str(target))
        else:
            target = root / f"{subject}.csv"  # 文件
            is_valid = target.exists() and target.is_file()
            call_kwargs = dict(mode=stage, file_path=str(target))

        if not is_valid:
            print(f"⚠️ 无效路径：{target}（跳过）")
            skip += 1
            continue

        print(f"\n=== ▶ 运行 {stage}: {target} ===")
        try:
            main.main(**call_kwargs)
            ok += 1
        except Exception as e:
            print(f"❌ 执行失败：{target} → {e}")

    print(f"\n🏁 阶段 {stage} 完成 | 成功: {ok} | 跳过: {skip}\n")


# ======================================================
# 根据 start/end 执行阶段序列
# ======================================================

def run_pipeline(subjects, root_base, start_stage, end_stage):
    stage_order = ["getmeta", "getref", "getinput", "getrank"]

    stage_roots = {
        "getmeta": "01meta_data",
        "getref": "02crossref_data",
        "getinput": "03openalex_data",
        "getrank": "04input_data",
    }

    if start_stage not in stage_order:
        raise ValueError(f"无效 start 阶段: {start_stage}")
    if end_stage not in stage_order:
        raise ValueError(f"无效 end 阶段: {end_stage}")

    start_idx = stage_order.index(start_stage)
    end_idx = stage_order.index(end_stage)

    if start_idx > end_idx:
        raise ValueError(f"--start ({start_stage}) 不能在 --end ({end_stage}) 之后")

    stages_to_run = stage_order[start_idx:end_idx+1]

    print("\n📌 将依次执行阶段：", " → ".join(stages_to_run), "\n")

    for stage in stages_to_run:
        root = Path(root_base) / stage_roots[stage]
        run_stage(subjects, stage, root)


# ======================================================
# CLI
# ======================================================

def cli():
    parser = argparse.ArgumentParser(description="批量运行 getmeta/getref/getinput/getrank 四阶段")
    parser.add_argument("--subjects", nargs="*", help="手动传入多个 subject")
    parser.add_argument("--list", type=str, help="从 txt 文件读取 subject 列表")
    parser.add_argument("--root", type=str, default="data",
                        help="根目录（包含 01meta_data 02crossref_data 等）")

    parser.add_argument("--start", type=str, default="getmeta",
                        help="起始阶段 getmeta/getref/getinput/getrank")
    parser.add_argument("--end", type=str, default="getrank",
                        help="结束阶段 getmeta/getref/getinput/getrank")

    args = parser.parse_args()

    # 读取学科列表
    if args.list:
        subjects, _ = load_subjects_from_txt(args.list)
    elif args.subjects:
        subjects = args.subjects
    else:
        print("❌ 请使用 --subjects 或 --list 指定学科列表")
        sys.exit(1)

    if not subjects:
        print("⚠️ 没有任何学科需要运行")
        sys.exit(0)

    # 执行流水线
    run_pipeline(subjects, root_base=args.root,
                 start_stage=args.start, end_stage=args.end)


if __name__ == "__main__":
    cli()

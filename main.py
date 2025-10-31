# -*- coding: utf-8 -*-
"""
主程序入口 - 批量处理学科数据（支持单文件/单目录运行）
---------------------------------------------------------
新增功能：
  ✅ --file 参数：指定单个 CSV 文件进行处理（适用于 getref / getinput / getrank）
  ✅ --dir 参数：指定单个目录运行（适用于 getmeta 阶段）
  ✅ 自动判断路径类型（单文件 / 批量模式）
---------------------------------------------------------
"""

from utils.subject_calculator import SubjectCalculator
from tqdm import tqdm
import os
import sys
from pathlib import Path
from data.scripts.get_by_crossref_openalex import CrossrefMetaProcessor
import json
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

try:
    from data.scripts.get_ref_openalex import RefOpenAlexMapper
except Exception as e:
    RefOpenAlexMapper = None


# ====================== 基础工具 ======================

def get_all_subdirectories(root_dir: str) -> list:
    """获取根目录下的所有子目录"""
    return sorted([os.path.join(root_dir, d) for d in os.listdir(root_dir)
                   if os.path.isdir(os.path.join(root_dir, d))])


def get_existing_files(directory: str) -> set:
    """获取目录中已有的CSV文件名集合"""
    if not os.path.exists(directory):
        return set()
    return {f for f in os.listdir(directory) if f.endswith(".csv")}


# ====================== 核心阶段函数 ======================

def get_crossref_openalex(input_dir: str, base_output_dir="data/02crossref_data", limit=3000):
    """执行 Crossref + OpenAlex 抓取"""
    dir_name = os.path.basename(input_dir)
    print(f"🔍 处理目录: {dir_name}")

    try:
        processor = CrossrefMetaProcessor(input_dir=input_dir, output_dir=base_output_dir)
        output_file = processor.merge_metadata_with_crossref(limit=limit)
        processor.print_statistics()
        print(f"✅ 处理完成: {dir_name}")
        return output_file
    except Exception as e:
        import traceback
        print(f"❌ 出错: {dir_name} → {e}")
        traceback.print_exc()
        return None


def batch_make_all_inputs(openalex_dir="data/03openalex_data",
                          mapping_csv="data/zh_disciplines.csv",
                          output_dir="data/04input_data",
                          file_path=None):
    """生成统一输入文件"""
    from data.scripts.make_input import make_all_lists
    os.makedirs(output_dir, exist_ok=True)

    if file_path:
        files = [os.path.basename(file_path)]
        openalex_dir = os.path.dirname(file_path)
    else:
        files = sorted([f for f in os.listdir(openalex_dir) if f.endswith(".csv")])

    print(f"📚 共 {len(files)} 个文件待处理")
    for fname in files:
        input_path = os.path.join(openalex_dir, fname)
        output_path = os.path.join(output_dir, fname)
        print(f"➡️ {fname}")
        if os.path.exists(output_path):
            print("   ⏭️ 已存在，跳过")
            continue
        try:
            make_all_lists(input_path, mapping_csv, output_path)
            print("   ✅ 成功生成")
        except Exception as e:
            import traceback
            print(f"   ❌ 失败: {e}")
            traceback.print_exc()


def refrank_with_llm_and_stub_result(
    input_dir="data/04input_data",
    output_dir="data/05output_data",
    overwrite=False,
    file_path=None,
):
    """阶段四：直接计算学科结果（可指定单个文件）"""
    os.makedirs(output_dir, exist_ok=True)
    files = [os.path.basename(file_path)] if file_path else sorted(
        [f for f in os.listdir(input_dir) if f.endswith(".csv")]
    )
    if file_path:
        input_dir = os.path.dirname(file_path)

    print(f"🧠 学科计算阶段：共 {len(files)} 个文件")
    calc = SubjectCalculator(debug=True, strategy="weighted",topk_cross=3)

    for fname in files:
        in_csv = os.path.join(input_dir, fname)
        out_csv = os.path.join(output_dir, fname)
        print(f"\n➡️ {fname}")

        if not overwrite and os.path.exists(out_csv):
            print("   ⏭️ 已存在，跳过")
            continue

        try:
            df = pd.read_csv(in_csv, dtype=str).fillna("")
        except Exception as e:
            print(f"   ❌ 读取失败：{e}")
            continue

        results_json = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="  ⚙️ 计算", ncols=90, leave=False):
            try:
                res = calc.calc(row)
                results_json.append(json.dumps(res, ensure_ascii=False))
            except Exception:
                results_json.append("{}")

        df["result"] = results_json
        df["primary"] = df["result"].apply(lambda x: json.loads(x).get("primary"))
        df["cross"] = df["result"].apply(lambda x: ",".join(json.loads(x).get("cross", [])))
        df["detail"] = df["result"].apply(
            lambda x: json.dumps(json.loads(x).get("detail", {}), ensure_ascii=False)
        )

        try:
            df.to_csv(out_csv, index=False, encoding="utf-8-sig")
            print(f"   💾 保存完成: {out_csv}")
        except Exception as e:
            print(f"   ❌ 写出失败: {e}")


# ====================== 主控制入口 ======================

def main(mode="all", file_path=None, dir_path=None):
    """主程序入口（支持单文件/单目录运行）"""
    ROOT_DIR = "data/01meta_data"
    PROCESSED_DIR = "data/02crossref_data"
    OPENALEX_REF_DIR = "data/03openalex_data"
    RESULT_DIR = "data/05output_data"

    print("=" * 60)
    print(f"🚀 启动批处理程序  | 模式: {mode}")
    print("=" * 60)

    # ---------------- getmeta ----------------
    if mode in ("getmeta", "all"):
        print("\n🧩 阶段 1：Crossref & OpenAlex 元数据获取")
        targets = [dir_path] if dir_path else get_all_subdirectories(ROOT_DIR)
        for i, subdir in enumerate(targets, 1):
            print(f"[{i}/{len(targets)}] {os.path.basename(subdir)}")
            get_crossref_openalex(subdir, PROCESSED_DIR)

    # ---------------- getref ----------------
    if mode in ("getref", "all"):
        if RefOpenAlexMapper is None:
            print("❌ 缺少 RefOpenAlexMapper 实现文件")
            return 1

        print("\n📚 阶段 2：参考文献 OpenAlex 映射")
        files = [os.path.basename(file_path)] if file_path else sorted(get_existing_files(PROCESSED_DIR))
        input_dir = os.path.dirname(file_path) if file_path else PROCESSED_DIR
        for fname in files:
            input_csv = os.path.join(input_dir, fname)
            output_csv = os.path.join(OPENALEX_REF_DIR, fname)
            print(f"➡️ {fname}")
            if os.path.exists(output_csv):
                print("   ⏭️ 已存在，跳过")
                continue
            mapper = RefOpenAlexMapper(input_csv, output_dir=OPENALEX_REF_DIR)
            mapper.process_ref_openalex(max_ref_per_paper=20, max_workers=10)
            mapper.print_statistics()
            print("   ✅ 完成")

    # ---------------- getinput ----------------
    if mode in ("getinput", "all"):
        print("\n🧩 阶段 3：生成统一输入文件")
        batch_make_all_inputs(
            openalex_dir="data/03openalex_data",
            mapping_csv="data/zh_disciplines.csv",
            output_dir="data/04input_data",
            file_path=file_path,
        )

    # ---------------- getrank ----------------
    if mode in ("getrank", "all"):
        print("\n🤖 阶段 4：学科计算")
        refrank_with_llm_and_stub_result(
            input_dir="data/04input_data",
            output_dir="data/05output_data",
            overwrite=False,
            file_path=file_path,
        )

    print("\n🎉 任务完成！")
    return 0


# ====================== CLI 调用 ======================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="批量处理学科数据（支持单文件或单目录运行）")
    parser.add_argument("--mode",
                        choices=["getmeta", "getref", "getinput", "getrank", "all"],
                        default="all",
                        help="执行阶段")
    parser.add_argument("--file", type=str, default=None,
                        help="指定单个CSV文件（用于 getref/getinput/getrank）")
    parser.add_argument("--dir", type=str, default=None,
                        help="指定单个目录（用于 getmeta）")

    args = parser.parse_args()
    exit(main(mode=args.mode, file_path=args.file, dir_path=args.dir))

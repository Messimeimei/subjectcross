# -*- coding: utf-8 -*-
"""
主程序入口 - 批量处理学科数据
功能：
1. 遍历data/meta_data下的所有学科目录
2. 对每个学科目录调用CrossrefMetaProcessor处理元数据
3. 对处理后的CSV文件调用batch_csv_runner进行学科分类
4. 智能跳过已处理的文件
"""

import os
import sys
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

# 添加当前目录到Python路径，确保可以导入模块
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from data.scripts.get_by_crossref import CrossrefMetaProcessor
from pipeline.batch_score import process_csv_in_batches, save_results_to_csv


def get_all_subdirectories(root_dir: str) -> list:
    """获取根目录下的所有子目录"""
    subdirs = []
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):
            subdirs.append(item_path)
    return sorted(subdirs)  # 按字母顺序排序


def get_existing_files(directory: str) -> set:
    """获取目录中已存在的CSV文件集合（不含路径）"""
    existing_files = set()
    if os.path.exists(directory):
        for file in os.listdir(directory):
            if file.endswith('.csv'):
                existing_files.add(file)
    return existing_files


def needs_processing(dir_name: str, processed_dir: str, result_dir: str) -> tuple:
    """
    判断目录是否需要处理
    返回: (need_crossref_processing, need_classification_processing)
    """
    expected_filename = f"{dir_name}.csv"

    # 检查processed_data中是否已有文件
    processed_file = os.path.join(processed_dir, expected_filename)
    has_processed_file = os.path.exists(processed_file)

    # 检查result_data中是否已有文件
    result_file = os.path.join(result_dir, expected_filename)
    has_result_file = os.path.exists(result_file)

    # 逻辑：
    # 1. 如果result_data中有文件，说明已经完成学科分类，都不需要处理
    if has_result_file:
        return False, False

    # 2. 如果processed_data中有文件但result_data中没有，只需要学科分类
    if has_processed_file:
        return False, True

    # 3. 如果两个目录都没有文件，都需要处理
    return True, True


def process_single_directory(input_dir: str, base_output_dir: str = "data/02crossref_data") -> str:
    """处理单个目录：获取Crossref元数据并保存为CSV"""
    dir_name = os.path.basename(input_dir)
    print(f"🔍 正在处理Crossref元数据: {dir_name}")

    try:
        processor = CrossrefMetaProcessor(input_dir, base_output_dir)
        output_file = processor.merge_metadata_with_crossref()
        processor.print_statistics()
        print(f"✅ Crossref处理完成: {dir_name}")
        return output_file
    except Exception as e:
        print(f"❌ Crossref处理目录 {dir_name} 时出错: {e}")
        return None


def process_single_classification(csv_file: str, result_base_dir: str = "data/03subject_data") -> str:
    """对单个CSV文件进行学科分类 - 按指定格式输出"""
    import json
    from pipeline.batch_score import DEFAULT_CSV_BATCH_SIZE

    csv_name = os.path.basename(csv_file)
    print(f"🎯 正在对 {csv_name} 进行学科分类...")

    try:
        out = process_csv_in_batches(csv_file, batch_size=DEFAULT_CSV_BATCH_SIZE)
        result_path = os.path.join(result_base_dir, csv_name)
        save_results_to_csv(out, result_path)

        print(f"✅ 学科分类完成: {csv_name}")

        return result_path
    except Exception as e:
        import traceback
        print(f"❌ 学科分类处理文件 {csv_file} 时出错: {e}")
        traceback.print_exc()
        return None



def batch_process_directories(root_dir: str, processed_base_dir: str = "data/02crossref_data",
                              result_base_dir: str = "data/03subject_data") -> dict:
    """
    批量处理所有目录，智能跳过已处理的文件
    返回处理统计信息
    """
    # 获取所有子目录
    subdirs = get_all_subdirectories(root_dir)
    print(f"📁 找到 {len(subdirs)} 个子目录需要处理")

    # 获取已存在的文件
    existing_processed_files = get_existing_files(processed_base_dir)
    existing_result_files = get_existing_files(result_base_dir)

    print(f"📊 发现 {len(existing_processed_files)} 个已处理的Crossref文件")
    print(f"📊 发现 {len(existing_result_files)} 个已完成的学科分类文件")

    # 统计信息
    stats = {
        'total_dirs': len(subdirs),
        'skipped_crossref': 0,
        'processed_crossref': 0,
        'skipped_classification': 0,
        'processed_classification': 0,
        'crossref_errors': 0,
        'classification_errors': 0,
        'processed_files': [],
        'skipped_dirs': []
    }

    # 处理每个目录
    for i, subdir in enumerate(subdirs, 1):
        dir_name = os.path.basename(subdir)
        print(f"\n[{i}/{len(subdirs)}] 处理目录: {dir_name}")

        # 判断需要哪些处理
        need_crossref, need_classification = needs_processing(dir_name, processed_base_dir, result_base_dir)

        if not need_crossref and not need_classification:
            print("   ⏭️  完全跳过（已存在最终结果）")
            stats['skipped_dirs'].append(dir_name)
            stats['skipped_crossref'] += 1
            stats['skipped_classification'] += 1
            continue

        processed_file_path = None

        # Crossref元数据处理
        if need_crossref:
            output_file = process_single_directory(subdir, processed_base_dir)
            if output_file and os.path.exists(output_file):
                processed_file_path = output_file
                stats['processed_crossref'] += 1
                print(f"   ✅ Crossref处理完成")
            else:
                stats['crossref_errors'] += 1
                print(f"   ❌ Crossref处理失败")
                continue
        else:
            # 使用已存在的processed文件
            processed_file_path = os.path.join(processed_base_dir, f"{dir_name}.csv")
            if os.path.exists(processed_file_path):
                print(f"   ⏭️  跳过Crossref处理（使用已有文件）")
                stats['skipped_crossref'] += 1
            else:
                print(f"   ⚠️  预期找到processed文件但不存在: {processed_file_path}")
                continue

        # 学科分类处理
        if need_classification and processed_file_path:
            result_file = process_single_classification(processed_file_path, result_base_dir)
            if result_file and os.path.exists(result_file):
                stats['processed_classification'] += 1
                stats['processed_files'].append(result_file)
                print(f"   ✅ 学科分类完成")
            else:
                stats['classification_errors'] += 1
                print(f"   ❌ 学科分类失败")
        else:
            print(f"   ⏭️  跳过学科分类（已存在结果）")
            stats['skipped_classification'] += 1

    return stats


def check_environment():
    """检查必要的环境和文件"""
    required_dirs = [
        "data/01meta_data",
        "data/02crossref_data",
        "data/03subject_data"
    ]

    required_files = [
        "data/zh_disciplines_with_code.csv",
        "data/zh_discipline_intro_with_code.json"
    ]

    print("🔍 检查环境配置...")

    # 检查目录
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"⚠️  创建目录: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
        else:
            print(f"✅ 目录存在: {dir_path}")

    # 检查文件
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ 文件不存在: {file_path}")
            return False
        else:
            print(f"✅ 文件存在: {file_path}")

    # 检查模型目录
    if not os.path.exists("models/bge-m3"):
        print("❌ 模型目录不存在: models/bge-m3")
        return False
    else:
        print("✅ 模型目录存在: models/bge-m3")

    return True


def print_final_statistics(stats: dict, result_dir: str):
    """打印最终统计信息"""
    print("\n" + "=" * 60)
    print("📊 最终处理统计")
    print("=" * 60)

    print(f"📁 总目录数: {stats['total_dirs']}")
    print()

    print(f"🔍 Crossref元数据处理:")
    print(f"   - 新处理: {stats['processed_crossref']} 个")
    print(f"   - 跳过: {stats['skipped_crossref']} 个")
    print(f"   - 错误: {stats['crossref_errors']} 个")
    print()

    print(f"🎯 学科分类处理:")
    print(f"   - 新处理: {stats['processed_classification']} 个")
    print(f"   - 跳过: {stats['skipped_classification']} 个")
    print(f"   - 错误: {stats['classification_errors']} 个")
    print()

    # 显示跳过的目录
    if stats['skipped_dirs']:
        print(f"⏭️  完全跳过的目录 ({len(stats['skipped_dirs'])} 个):")
        for dir_name in stats['skipped_dirs']:
            print(f"   - {dir_name}")
        print()

    # 显示总文件统计
    total_result_files = len([f for f in os.listdir(result_dir) if f.endswith('.csv')])
    print(f"📦 结果目录中现有文件: {total_result_files} 个")

    # 显示总记录数
    total_records = 0
    for result_file in os.listdir(result_dir):
        if result_file.endswith('.csv'):
            try:
                import pandas as pd
                df = pd.read_csv(os.path.join(result_dir, result_file))
                total_records += len(df)
                print(f"   - {result_file}: {len(df)} 条记录")
            except:
                pass

    print(f"📄 总论文记录数: {total_records} 条")


def main():
    """主函数"""
    # 配置路径 - 根据你的实际目录结构调整
    ROOT_DIR = "data/01meta_data"  # 包含多个学科目录的根目录
    PROCESSED_DIR = "data/02crossref_data"  # Crossref处理结果目录
    RESULT_DIR = "data/03subject_data"  # 最终学科分类结果目录

    print("🚀 开始批量处理学科数据...")
    print("=" * 60)

    # 检查环境
    if not check_environment():
        print("💥 环境检查失败，请确保所有必要文件和目录存在")
        return 1

    # 检查是否有学科目录需要处理
    subdirs = get_all_subdirectories(ROOT_DIR)
    if not subdirs:
        print("❌ 没有找到任何学科目录，请检查 data/01meta_data 目录")
        return 1

    print(f"\n📚 找到 {len(subdirs)} 个学科目录:")
    for subdir in subdirs:
        print(f"   - {os.path.basename(subdir)}")

    try:
        # 批量处理所有目录
        stats = batch_process_directories(ROOT_DIR, PROCESSED_DIR, RESULT_DIR)

        # 打印最终统计
        print_final_statistics(stats, RESULT_DIR)

        # 检查是否有错误
        if stats['crossref_errors'] > 0 or stats['classification_errors'] > 0:
            print(f"\n⚠️  注意：处理过程中发生了 {stats['crossref_errors'] + stats['classification_errors']} 个错误")
            return 1
        else:
            print(f"\n🎉 所有处理完成！")
            return 0

    except Exception as e:
        print(f"💥 程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
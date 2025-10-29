# -*- coding: utf-8 -*-
"""
生成统一输入文件（融合Qwen机构识别版）：
从含 openalex 数据的 csv 中提取并生成多维度一级学科列表：
- list_incites_direction：研究方向映射结果（附均分得分）
- list_title_abs：标题+摘要相似度TopN学科（向量法）
- list_author_aff_qwen：作者+机构 → Qwen判定（每机构最多2个学科）
- list_openalex：直接复制 OpenAlex_map_subjects 列
- list_ref：参考文献TF-IDF权重学科列表
"""

import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import os, ast, json, time
import pandas as pd
import numpy as np
from collections import Counter
from typing import List, Tuple, Dict
from tqdm import tqdm
from dotenv import load_dotenv
from utils.vector2discipline import VectorDisciplineScorer, cache_path
from utils.llm_call import call_qwen_rank

load_dotenv()

# ========= 环境变量 =========
EMB_MODEL_NAME = os.getenv("EMB_MODEL_NAME", "../../models/bge-m3")
CSV_PATH = os.getenv("CSV_PATH", "../zh_disciplines.csv")
JSON_PATH = os.getenv("JSON_PATH", "../zh_discipline_intro.json")


# ---------- 研究方向 ----------
def add_incites_direction_list(df: pd.DataFrame, mapping_csv: str, direction_col: str = "研究方向") -> pd.Series:
    """
    将研究方向映射为学科列表，并给每个学科分配均等分数 1/N
    输出示例：
      [('0812 计算机科学与技术', 0.25), ('0835 软件工程', 0.25)]
    """
    mapping_df = pd.read_csv(mapping_csv, header=None, names=["raw"], dtype=str).fillna("")
    code2name = {}
    for x in mapping_df["raw"]:
        x = x.strip()
        if len(x) >= 5 and x[:4].isdigit():
            code2name[x[:4]] = x[5:].strip()

    def parse_direction(direction_str: str):
        if not direction_str or not isinstance(direction_str, str):
            return []
        parts = [p.strip() for p in direction_str.split(";") if p.strip()]
        result = []
        for p in parts:
            code = p.split()[0]
            if code.isdigit() and len(code) == 4 and code in code2name:
                result.append(f"{code} {code2name[code]}")
        if not result:
            return []
        score = round(1 / len(result), 4)
        return [(r, score) for r in result]

    return df[direction_col].apply(parse_direction)

# ---------- OpenAlex ----------
def add_openalex_list(df: pd.DataFrame, col: str = "OpenAlex_map_subjects", topk: int = 5) -> pd.Series:
    """
    对OpenAlex字段的学科列表进行统一化计算（先使用新映射表重新映射）：
    1. 使用新的映射表重新映射 OpenAlex_field_list 和 OpenAlex_subfield_list
    2. 将每篇论文的OpenAlex子学科集合整合为一级学科分布；
    3. 每个学科的得分 = 在所有子列表中出现时的平均得分；
    4. 去掉稀有度惩罚，仅保留平均强度；
    5. 最终做softmax归一化输出前TopK结果。
    """
    # ========== 第一步：加载新的映射表 ==========
    mapping_file = "data/deepseek_map.json"
    openalex_to_cn = {}
    if os.path.exists(mapping_file):
        try:
            with open(mapping_file, "r", encoding="utf-8") as f:
                openalex_to_cn = json.load(f)
            print(f"✅ 已加载新的OpenAlex映射表：{mapping_file}")
        except Exception as e:
            print(f"⚠️ 加载映射表失败: {e}")
    else:
        print(f"⚠️ 未找到映射文件: {mapping_file}")

    def map_to_cn_groups(fields: List[str], subfields: List[str]) -> List[List[tuple]]:
        """
        为每个 field/subfield 生成独立映射列表
        每个元素为 (学科, 分数)
        """
        groups = []
        for name in (fields or []) + (subfields or []):
            mapped_pairs = openalex_to_cn.get(name, [])
            # 确保格式为 [["学科", 分数], ...]
            clean_pairs = []
            for m in mapped_pairs:
                if isinstance(m, (list, tuple)) and len(m) == 2:
                    subj, score = m
                    try:
                        clean_pairs.append((str(subj), float(score)))
                    except Exception:
                        continue
            groups.append(clean_pairs)
        return groups

    # ========== 第二步：重新映射OpenAlex数据 ==========
    def remap_openalex(row):
        """重新映射OpenAlex的field和subfield"""
        try:
            # 获取原始的field_list和subfield_list
            fields = row.get("OpenAlex_field_list", [])
            subfields = row.get("OpenAlex_subfield_list", [])

            # 如果是字符串，尝试解析为列表
            if isinstance(fields, str):
                fields = ast.literal_eval(fields) if fields.strip() else []
            if isinstance(subfields, str):
                subfields = ast.literal_eval(subfields) if subfields.strip() else []

            # 使用新映射表重新映射
            return map_to_cn_groups(fields, subfields)
        except Exception as e:
            print(f"⚠️ 重新映射OpenAlex失败: {e}")
            return []

    # 应用重新映射
    print("🔄 使用新映射表重新映射OpenAlex数据...")
    df["OpenAlex_map_subjects_remapped"] = df.apply(remap_openalex, axis=1)

    # ========== 第三步：原有的统一化计算（使用重新映射后的数据） ==========
    def aggregate_openalex(subj_list):
        if not subj_list or not isinstance(subj_list, list):
            return []
        try:
            refs = [r for r in subj_list if isinstance(r, list) and r]
            if len(refs) == 0:
                return []

            score_sum = Counter()
            doc_freq = Counter()

            for ref in refs:
                seen_subjs = set()
                for subj, score in ref:
                    score_sum[subj] += score
                    seen_subjs.add(subj)
                for s in seen_subjs:
                    doc_freq[s] += 1

            # 平均得分（无稀有度）
            final_scores = {subj: score_sum[subj] / doc_freq[subj] for subj in score_sum.keys()}

            # softmax归一化
            vals = np.array(list(final_scores.values()), dtype=float)
            e_x = np.exp(vals - np.max(vals))
            probs = e_x / e_x.sum()

            items = sorted(zip(final_scores.keys(), probs), key=lambda x: x[1], reverse=True)[:topk]
            return [(k, round(float(v), 4)) for k, v in items]

        except Exception:
            return []

    # 使用重新映射后的数据进行统一化计算
    result = df["OpenAlex_map_subjects_remapped"].apply(aggregate_openalex)

    # 清理临时列
    if "OpenAlex_map_subjects_remapped" in df.columns:
        df.drop(columns=["OpenAlex_map_subjects_remapped"], inplace=True)

    return result


# ---------- 参考文献 ----------
def add_ref_list(df: pd.DataFrame, ref_col: str = "Ref_OpenAlex_map_subjects", topk: int = 10) -> pd.Series:
    """
    从参考文献中提取整体学科分布（先使用新映射表重新映射）：
    - 使用新的映射表重新映射参考文献的OpenAlex数据
    - 高频出现的学科代表研究主线；
    - 每个学科的得分为该学科在所有参考文献中的平均得分；
    - 最终用 softmax 归一化为概率分布。
    """
    # ========== 第一步：加载新的映射表 ==========
    mapping_file = "data/deepseek_map.json"
    openalex_to_cn = {}
    if os.path.exists(mapping_file):
        try:
            with open(mapping_file, "r", encoding="utf-8") as f:
                openalex_to_cn = json.load(f)
            print(f"✅ 已加载新的OpenAlex映射表（参考文献）：{mapping_file}")
        except Exception as e:
            print(f"⚠️ 加载映射表失败: {e}")
    else:
        print(f"⚠️ 未找到映射文件（参考文献）: {mapping_file}")

        # 如果找不到映射文件，使用原始的逻辑
        def aggregate_original_ref_subjects(ref_str):
            if not ref_str or not isinstance(ref_str, str):
                return []
            try:
                refs = ast.literal_eval(ref_str)
                refs = [r for r in refs if isinstance(r, list) and r]
                if len(refs) == 0:
                    return []

                score_sum = Counter()
                doc_freq = Counter()

                for ref in refs:
                    seen_subjs = set()
                    for subj, score in ref:
                        score_sum[subj] += score
                        seen_subjs.add(subj)
                    for s in seen_subjs:
                        doc_freq[s] += 1

                # ✅ 去掉稀有度，只按平均得分
                final_scores = {subj: score_sum[subj] / doc_freq[subj] for subj in score_sum.keys()}

                # softmax归一化
                vals = np.array(list(final_scores.values()), dtype=float)
                e_x = np.exp(vals - np.max(vals))
                probs = e_x / e_x.sum()

                items = sorted(zip(final_scores.keys(), probs), key=lambda x: x[1], reverse=True)[:topk]
                return [(k, round(float(v), 4)) for k, v in items]

            except Exception:
                return []

        return df[ref_col].apply(aggregate_original_ref_subjects)

    # ========== 第二步：重新映射参考文献数据 ==========
    def map_to_cn_groups(fields: List[str], subfields: List[str]) -> List[List[tuple]]:
        """
        为每个 field/subfield 生成独立映射列表
        每个元素为 (学科, 分数)
        """
        groups = []
        for name in (fields or []) + (subfields or []):
            mapped_pairs = openalex_to_cn.get(name, [])
            # 确保格式为 [["学科", 分数], ...]
            clean_pairs = []
            for m in mapped_pairs:
                if isinstance(m, (list, tuple)) and len(m) == 2:
                    subj, score = m
                    try:
                        clean_pairs.append((str(subj), float(score)))
                    except Exception:
                        continue
            groups.append(clean_pairs)
        return groups

    def remap_ref_topics(row):
        """重新映射参考文献的OpenAlex topics"""
        try:
            # 获取原始的参考文献topics
            ref_topics = row.get("Ref_OpenAlex_topics", [])

            # 如果是字符串，尝试解析为列表
            if isinstance(ref_topics, str):
                ref_topics = ast.literal_eval(ref_topics) if ref_topics.strip() else []

            remapped_refs = []
            for ref in ref_topics:
                if isinstance(ref, list) and len(ref) >= 2:
                    fields = ref[0] if isinstance(ref[0], list) else []
                    subfields = ref[1] if len(ref) > 1 and isinstance(ref[1], list) else []

                    # 使用新映射表重新映射
                    mapped_groups = map_to_cn_groups(fields, subfields)
                    remapped_refs.append(mapped_groups)

            return remapped_refs
        except Exception as e:
            print(f"⚠️ 重新映射参考文献失败: {e}")
            return []

    # 应用重新映射
    print("🔄 使用新映射表重新映射参考文献数据...")
    df["Ref_OpenAlex_map_subjects_remapped"] = df.apply(remap_ref_topics, axis=1)

    # ========== 第三步：原有的统一化计算（使用重新映射后的数据） ==========
    def aggregate_ref_subjects(ref_list):
        if not ref_list or not isinstance(ref_list, list):
            return []
        try:
            # 展平所有参考文献的映射结果
            all_refs = []
            for ref in ref_list:
                if isinstance(ref, list):
                    # 每个ref是一个参考文献的映射结果列表
                    for sublist in ref:
                        if isinstance(sublist, list):
                            all_refs.extend(sublist)

            if len(all_refs) == 0:
                return []

            score_sum = Counter()
            doc_freq = Counter()

            # 处理所有映射后的学科对
            for subj, score in all_refs:
                score_sum[subj] += score
                doc_freq[subj] += 1

            # ✅ 去掉稀有度，只按平均得分
            final_scores = {subj: score_sum[subj] / doc_freq[subj] for subj in score_sum.keys()}

            # softmax归一化
            vals = np.array(list(final_scores.values()), dtype=float)
            e_x = np.exp(vals - np.max(vals))
            probs = e_x / e_x.sum()

            items = sorted(zip(final_scores.keys(), probs), key=lambda x: x[1], reverse=True)[:topk]
            return [(k, round(float(v), 4)) for k, v in items]

        except Exception:
            return []

    # 使用重新映射后的数据进行统一化计算
    result = df["Ref_OpenAlex_map_subjects_remapped"].apply(aggregate_ref_subjects)

    # 清理临时列
    if "Ref_OpenAlex_map_subjects_remapped" in df.columns:
        df.drop(columns=["Ref_OpenAlex_map_subjects_remapped"], inplace=True)

    return result

# ---------- 标题+摘要 先相似度再大模型判断----------
def add_title_abs_scores(df: pd.DataFrame, topn: int = 5, use_gpu: bool = True) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    标题+摘要学科计算（双层模式+融合，支持并发+缓存）：
    ------------------------------------------------------------
    1. BGE 模型算语义相似度（全117学科）
    2. 调用 Qwen 模型判定（全117学科）
    3. 再次调用 Qwen 模型（仅输入 BGE 前5学科 + 简介）
    4. 并发执行、缓存已调用结果
    5. 输出：
        - list_title_abs_bge       向量模型结果
        - list_title_abs_qwen      全量Qwen结果
        - list_title_abs_ave       原融合结果（平均）
        - list_title_abs_merged    限定Qwen结果融合
        - list_title_abs           最终结果 = list_title_abs_merged
    ------------------------------------------------------------
    """

    # ========== Step 1. BGE 模型 ==========
    scorer = VectorDisciplineScorer(use_gpu=use_gpu)
    code2name, code2intro = scorer.load_disciplines()
    cpath = cache_path(EMB_MODEL_NAME, CSV_PATH, JSON_PATH)
    emb, codes, names, texts = scorer.ensure_cache(cpath, code2name, code2intro)

    text_titleabs = (df["论文标题"] + "。 " + df["CR_摘要"]).tolist()
    res_titleabs = scorer.score_batch(text_titleabs, codes, names, emb)

    list_bge_all = []
    for r in res_titleabs:
        topn_sorted = sorted(r.items(), key=lambda x: x[1], reverse=True)[:topn]
        list_bge_all.append([(k, float(v)) for k, v in topn_sorted])

    # ========== Step 2. 预加载学科列表 ==========
    print("🤖 准备 Qwen 学科信息...")
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        disciplines = [line.strip() for line in f if line.strip() and line[:4].isdigit()]

    # ========== Step 3. 缓存路径设置 ==========
    CACHE_DIR = "cache/qwen_titleabs"
    os.makedirs(CACHE_DIR, exist_ok=True)

    def cache_key(text: str, mode: str):
        """生成缓存文件路径"""
        h = hashlib.md5(text.encode("utf-8")).hexdigest()[:16]
        return os.path.join(CACHE_DIR, f"{mode}_{h}.json")

    def load_from_cache(text: str, mode: str):
        path = cache_key(text, mode)
        if os.path.exists(path):
            try:
                return json.load(open(path, "r", encoding="utf-8"))
            except:
                return None
        return None

    def save_to_cache(text: str, mode: str, data: dict):
        path = cache_key(text, mode)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ========== Step 4. 并发调用函数 ==========
    def qwen_call(text, bge_res, mode):
        """单次调用（带缓存）"""
        cache_res = load_from_cache(text, mode)
        if cache_res:
            return cache_res

        try:
            if mode == "all":
                res = call_qwen_rank(
                    text_block=text,
                    disciplines_json=disciplines,
                    disciplines_intro_json=code2intro,
                    topn=topn,
                    mode="title_abs"
                )
            else:
                local_disciplines = [k for k, _ in bge_res]
                local_intro = {k: code2intro.get(k, "") for k in local_disciplines}
                res = call_qwen_rank(
                    text_block=text,
                    disciplines_json=local_disciplines,
                    disciplines_intro_json=local_intro,
                    topn=topn,
                    mode="title_abs"
                )
            save_to_cache(text, mode, res)
            return res if res else bge_res
        except Exception as e:
            print(f"⚠️ Qwen调用失败 ({mode}): {e}")
            return bge_res

    # ========== Step 5. 并发执行 Qwen（全学科 + 局部学科）==========
    print("⚡ 并发调用 Qwen 模型（全学科 + 局部学科）...")
    list_qwen_all = [None] * len(df)
    list_qwen_local_all = [None] * len(df)

    with ThreadPoolExecutor(max_workers=10) as executor:  # 并发线程数可调
        futures = {}
        for idx, (text, bge_res) in enumerate(zip(text_titleabs, list_bge_all)):
            futures[executor.submit(qwen_call, text, bge_res, "all")] = ("all", idx)
            futures[executor.submit(qwen_call, text, bge_res, "limited")] = ("limited", idx)

        for fut in tqdm(as_completed(futures), total=len(futures), ncols=90):
            mode, i = futures[fut]
            res = fut.result()
            if mode == "all":
                list_qwen_all[i] = res
            else:
                list_qwen_local_all[i] = res

    # ========== Step 6. 融合逻辑 ==========
    def merge_results(bge_list, qwen_list):
        merged = {}
        for k, v in bge_list:
            merged[k] = merged.get(k, 0) + v
        for k, v in qwen_list:
            if k in merged:
                merged[k] = (merged[k] + v) / 2
            else:
                merged[k] = v
        merged_sorted = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:topn]
        return [(k, round(float(v), 4)) for k, v in merged_sorted]

    # 原融合结果（BGE+全学科Qwen）
    list_ave_all = [merge_results(bge, qwen) for bge, qwen in zip(list_bge_all, list_qwen_all)]
    # 新融合结果（BGE+局部Qwen）
    list_merged_all = [merge_results(bge, qwen_limited) for bge, qwen_limited in zip(list_bge_all, list_qwen_local_all)]

    # 最终结果 = 新融合
    list_final_all = list_merged_all

    print("✅ 学科计算完成！已启用并发+缓存加速。")

    return (
        pd.Series(list_bge_all),
        pd.Series(list_qwen_all),
        pd.Series(list_ave_all),       # 原平均融合
        pd.Series(list_merged_all),    # 局部限定融合
        pd.Series(list_final_all)      # 最终结果
    )



# ---------- 作者机构：直接调用 Qwen ----------
def add_author_aff_qwen(df: pd.DataFrame, topk: int = 5, topn_each: int = 2,
                        sleep_time: float = 0.4, max_workers: int = 10) -> pd.Series:
    """
    对每篇论文的作者机构字段并发调用 Qwen 模型：
    ----------------------------------------------------------
    - 每个机构名并发调用 Qwen 识别学科（每机构最多 topn_each 个）
    - 自动缓存已识别的机构结果（cache/qwen_author_aff/）
    - 对所有机构结果求平均后 softmax 归一化，取前 topk 输出
    输出格式：
      [('0812 计算机科学与技术', 0.45), ('0835 软件工程', 0.35), ...]
    ----------------------------------------------------------
    """

    # ✅ 从 CSV 读取学科列表
    disciplines = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if len(line) >= 5 and line[:4].isdigit():
                disciplines.append(line)

    # ✅ 缓存路径设置
    CACHE_DIR = "cache/qwen_author_aff"
    os.makedirs(CACHE_DIR, exist_ok=True)

    def cache_key(text: str):
        """生成缓存路径"""
        h = hashlib.md5(text.encode("utf-8")).hexdigest()[:16]
        return os.path.join(CACHE_DIR, f"{h}.json")

    def load_cache(text: str):
        path = cache_key(text)
        if os.path.exists(path):
            try:
                return json.load(open(path, "r", encoding="utf-8"))
            except:
                return None
        return None

    def save_cache(text: str, data: list):
        path = cache_key(text)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ✅ 单次调用函数
    def qwen_call_aff(aff: str):
        """单个机构调用（带缓存）"""
        cached = load_cache(aff)
        if cached is not None:
            return aff, cached

        try:
            mapped = call_qwen_rank(
                text_block=aff,
                disciplines_json=disciplines,
                disciplines_intro_json={},
                topn=topn_each,
                mode="author_aff"
            )
            save_cache(aff, mapped or [])
            time.sleep(sleep_time)
            return aff, mapped or []
        except Exception as e:
            print(f"⚠️ {aff} 调用失败：{e}")
            save_cache(aff, [])
            return aff, []

    # ✅ 收集所有机构名称（去重）
    all_affs = set()
    for aff_json_str in df["CR_作者和机构"].fillna("").tolist():
        try:
            aff_list = json.loads(aff_json_str)
            for author in aff_list:
                for aff in author.get("affiliation", []):
                    if isinstance(aff, str) and aff.strip():
                        all_affs.add(aff.strip())
        except Exception:
            continue
    all_affs = list(all_affs)

    print(f"⚡ 并发调用 Qwen 模型进行机构学科识别（共 {len(all_affs)} 个唯一机构）...")

    # ✅ 并发执行
    cache: Dict[str, List[Tuple[str, float]]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(qwen_call_aff, aff): aff for aff in all_affs}
        for fut in tqdm(as_completed(futures), total=len(futures), ncols=90):
            aff, mapped = fut.result()
            cache[aff] = mapped or []

    # ✅ 聚合每篇论文的机构结果
    results_all = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="聚合作者机构结果", ncols=100):
        aff_json_str = row.get("CR_作者和机构", "")
        if not aff_json_str:
            results_all.append([])
            continue

        try:
            aff_list = json.loads(aff_json_str)
        except Exception:
            results_all.append([])
            continue

        score_sum = Counter()
        doc_freq = Counter()

        for author in aff_list:
            for aff in author.get("affiliation", []):
                if not isinstance(aff, str) or not aff.strip():
                    continue
                aff_clean = aff.strip()
                mapped = cache.get(aff_clean, [])
                for subj, score in mapped or []:
                    score_sum[subj] += score
                    doc_freq[subj] += 1

        if not score_sum:
            results_all.append([])
            continue

        # 平均 + softmax 归一化
        avg_scores = {k: score_sum[k] / doc_freq[k] for k in score_sum.keys()}
        vals = np.array(list(avg_scores.values()), dtype=float)
        e_x = np.exp(vals - np.max(vals))
        probs = e_x / e_x.sum()
        normed = {k: float(v) for k, v in zip(avg_scores.keys(), probs)}

        top_items = sorted(normed.items(), key=lambda x: x[1], reverse=True)[:topk]
        results_all.append([(k, round(v, 4)) for k, v in top_items])

    print("✅ 作者机构学科识别完成！（已启用并发+缓存加速）")
    return pd.Series(results_all)

# ---------- 主函数 ----------
def make_all_lists(csv_path: str, mapping_csv: str, save_path: str = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str).fillna("")

    # ====== Step 1. 各字段生成 ======
    df["list_incites_direction"] = add_incites_direction_list(df, mapping_csv)
    df["list_ref"] = add_ref_list(df)
    df["list_title_abs_bge"], \
        df["list_title_abs_qwen"], \
        df["list_title_abs_ave"], \
        df["list_title_abs_merged"], \
        df["list_title_abs"] = add_title_abs_scores(df)

    df["list_author_aff_qwen"] = add_author_aff_qwen(df, topk=2)

    if "CR_出版商" in df.columns:
        df = df.drop(columns=["CR_出版商"])
    if "OpenAlex_map_subjects" in df.columns:
        df["list_openalex"] = add_openalex_list(df, col="OpenAlex_map_subjects")

    # ====== Step 2. 过滤：关键五列必须都有值 ======
    key_cols = [
        "list_incites_direction",
        "list_title_abs",
        "list_author_aff_qwen",
        "list_openalex",
        "list_ref"
    ]

    def _not_empty(v):
        """判断字段是否非空"""
        if v is None:
            return False
        if isinstance(v, str):
            v = v.strip()
            # 空字符串或空列表/字典表示空
            if v in ("", "[]", "{}", "null", "None"):
                return False
        if isinstance(v, (list, dict)) and len(v) == 0:
            return False
        return True

    before = len(df)
    df = df[df[key_cols].applymap(_not_empty).all(axis=1)]
    after = len(df)
    print(f"🧹 已过滤空值记录：{before - after} 条（保留 {after} 条完整记录）")

    # ====== Step 3. 输出 ======
    target_cols = [
        "list_title_abs_bge",
        "list_incites_direction",
        "list_title_abs",
        "list_author_aff_qwen",
        "list_openalex",
        "list_ref"
    ]
    cols = [c for c in df.columns if c not in target_cols] + target_cols
    df = df[cols]

    if save_path:
        df.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"✅ 已保存到 {save_path}")
    return df


# ---------- 示例 ----------
if __name__ == "__main__":
    pd.set_option("display.max_colwidth", None)
    output = make_all_lists(
        "../03openalex_data/0101 Philosophy.csv",
        "../zh_disciplines.csv",
        "../04input_data/0101 Philosophy.csv"
    )
    print(output.tail(3)[["list_incites_direction"]])

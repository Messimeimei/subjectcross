import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Tuple, Dict
from dotenv import load_dotenv
import pandas as pd
import os
import glob
from typing import List, Tuple
import json
# 加载环境变量
load_dotenv()

class QwenDisciplineScorer:
    """
    使用Qwen3-Embedding模型进行学科分类（内存优化版）
    """
    
    def __init__(self, model_path: str = None, use_flash_attention: bool = False, device: str = None):
        """
        初始化模型
        
        :param model_path: 模型路径，None则从环境变量读取
        :param use_flash_attention: 是否使用flash attention加速（默认关闭以减少内存）
        :param device: 设备类型，None则自动检测
        """
        # 从环境变量读取配置
        self.model_path = model_path or os.getenv("EMB_MODEL_NAME", "../../models/Qwen3-Embedding-0.6B")
        self.batch_size = int(os.getenv("BATCH_SIZE", "16"))  # 减小批大小
        self.use_fp16 = os.getenv("USE_FP16", "false").lower() == "true"  # 默认关闭FP16
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🤖 加载Qwen3-Embedding模型: {self.model_path}")
        print(f"⚙️ 配置 - 设备: {self.device}, 批大小: {self.batch_size}, FP16: {self.use_fp16}")
        
        # 检查模型路径是否存在
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型路径不存在: {self.model_path}")
        
        # 加载tokenizer和模型
        print("📥 加载tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            trust_remote_code=True,
            padding_side='left'
        )
        
        print("📥 加载模型...")
        try:
            # 使用更保守的设置以减少内存使用
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.use_fp16 else torch.float32,
                "low_cpu_mem_usage": True,
            }
            
            # 只在明确要求且内存充足时使用flash attention
            if use_flash_attention and torch.cuda.is_available():
                try:
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    print("⚡ 使用Flash Attention 2")
                except Exception as e:
                    print(f"⚠️ Flash Attention 2不可用: {e}")
            
            self.model = AutoModel.from_pretrained(self.model_path, **model_kwargs)
            
            # 移动到设备
            if self.device == "cuda":
                self.model = self.model.cuda()
            else:
                self.model = self.model.to(self.device)
                
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
            
        self.max_length = 2048  # 减小最大长度
        
        # 学科任务描述
        self.task_description = "给定一篇学术论文的标题和摘要，判断其所属的学科领域"
        
        # 加载学科信息
        self.code2name, self.code2intro = self.load_disciplines()
        
    def last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """
        使用last token pooling获取句子表示
        """
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    
    def get_detailed_instruct(self, query: str) -> str:
        """
        构建指令格式
        """
        return f'Instruct: {self.task_description}\nQuery: {query}'
    
    def load_disciplines(self) -> Tuple[Dict, Dict]:
        """
        加载学科信息，从环境变量指定的路径
        """
        json_path = os.getenv("JSON_PATH", "../zh_discipline_intro.json")
        csv_path = os.getenv("CSV_PATH", "../zh_disciplines.csv")
        
        # 确保路径是绝对路径
        if not os.path.isabs(json_path):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(current_dir, json_path)
            csv_path = os.path.join(current_dir, csv_path)
            
        print(f"📚 加载学科数据: {json_path}")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                discipline_data = json.load(f)
        except Exception as e:
            print(f"❌ 加载学科JSON文件失败: {e}")
            return self._load_disciplines_from_csv(csv_path)
        
        code2name = {}
        code2intro = {}
        
        for code, info in discipline_data.items():
            code2name[code] = info.get('name', '')
            code2intro[code] = info.get('intro', '')
            
        print(f"✅ 从JSON加载了 {len(code2name)} 个学科")
        return code2name, code2intro
    
    def _load_disciplines_from_csv(self, csv_path: str) -> Tuple[Dict, Dict]:
        """
        从CSV文件加载学科信息（备用方案）
        """
        try:
            import pandas as pd
            df = pd.read_csv(csv_path, header=None, names=["raw"])
            code2name = {}
            code2intro = {}
            
            for x in df["raw"]:
                x = str(x).strip()
                if len(x) >= 5 and x[:4].isdigit():
                    code = x[:4]
                    name = x[5:].strip()
                    code2name[code] = name
                    code2intro[code] = name
            
            print(f"✅ 从CSV加载了 {len(code2name)} 个学科")
            return code2name, code2intro
        except Exception as e:
            print(f"❌ 从CSV加载学科也失败: {e}")
            return {}, {}
    
    def prepare_discipline_texts(self, code2name: Dict, code2intro: Dict, max_intro_length: int = 2000) -> List[str]:
        """
        准备学科文本：代码 + 名称 + 介绍（截断过长的介绍）
        """
        discipline_texts = []
        for code, name in code2name.items():
            intro = code2intro.get(code, "")
            # 截断过长的介绍
            if len(intro) > max_intro_length:
                intro = intro[:max_intro_length] + "..."
            # 构建学科描述文本
            text = f"{code} {name}。{intro}"
            discipline_texts.append(text)
        return discipline_texts
    
    def get_embeddings_batch(self, texts: List[str]) -> Tensor:
        """
        分批获取文本的嵌入向量（内存优化）
        """
        if not texts:
            return torch.tensor([]).to(self.device)
        
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            
            # Tokenize
            batch_dict = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
            
            # 模型推理
            with torch.no_grad():
                outputs = self.model(**batch_dict)
            
            # Last token pooling
            embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            
            # 归一化
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.cpu())  # 移到CPU释放GPU内存
            
            # 清理GPU缓存
            if self.device == "cuda":
                torch.cuda.empty_cache()
        
        # 合并所有批次的嵌入
        if all_embeddings:
            return torch.cat(all_embeddings, dim=0).to(self.device)
        else:
            return torch.tensor([]).to(self.device)
    
    def score_single_memory_efficient(self, title: str, abstract: str, topk: int = None) -> List[Tuple[str, float]]:
        """
        内存优化的单篇论文评分（分批处理学科）
        """
        topk = topk or int(os.getenv("TOPN", "5"))
        
        if not self.code2name:
            return []
        
        # 准备查询文本
        query_text = f"标题：{title}。摘要：{abstract}"
        instructed_query = self.get_detailed_instruct(query_text)
        
        # 获取查询嵌入
        print("🔍 计算查询嵌入...")
        query_embedding = self.get_embeddings_batch([instructed_query])
        if query_embedding.numel() == 0:
            return []
        
        # 分批处理学科文本
        discipline_texts = self.prepare_discipline_texts(self.code2name, self.code2intro)
        discipline_codes = list(self.code2name.keys())
        
        all_scores = []
        
        print("📚 分批计算学科相似度...")
        for i in range(0, len(discipline_texts), self.batch_size):
            batch_texts = discipline_texts[i:i+self.batch_size]
            batch_codes = discipline_codes[i:i+self.batch_size]
            
            # 获取当前批次的学科嵌入
            discipline_embeddings = self.get_embeddings_batch(batch_texts)
            if discipline_embeddings.numel() == 0:
                continue
            
            # 计算相似度
            batch_scores = (query_embedding @ discipline_embeddings.T).squeeze(0)
            batch_scores = batch_scores.cpu().numpy()
            
            # 保存分数和对应的学科代码
            for j, score in enumerate(batch_scores):
                all_scores.append((batch_codes[j], float(score)))
            
            # 清理GPU缓存
            if self.device == "cuda":
                torch.cuda.empty_cache()
        
        # 按分数排序并返回topk
        all_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 转换为包含学科名称的格式
        formatted_results = []
        for code, score in all_scores[:topk]:
            name = self.code2name.get(code, "")
            formatted_results.append((f"{code} {name}", score))
        
        return formatted_results
    
    def score_batch_memory_efficient(self, titles: List[str], abstracts: List[str], topk: int = None, query_batch_size: int = None) -> List[List[Tuple[str, float]]]:
        """
        内存优化的批量处理
        """
        topk = topk or int(os.getenv("TOPN", "5"))
        query_batch_size = query_batch_size or max(1, self.batch_size // 4)  # 更小的查询批大小
        
        if not self.code2name:
            return [[] for _ in range(len(titles))]
        
        # 预计算所有学科嵌入（分批）
        print("📚 预计算学科嵌入...")
        discipline_texts = self.prepare_discipline_texts(self.code2name, self.code2intro)
        discipline_codes = list(self.code2name.keys())
        discipline_embeddings = self.get_embeddings_batch(discipline_texts)
        
        all_results = []
        
        # 分批处理查询
        for i in range(0, len(titles), query_batch_size):
            batch_titles = titles[i:i+query_batch_size]
            batch_abstracts = abstracts[i:i+query_batch_size]
            
            print(f"🔢 处理查询批次 {i//query_batch_size + 1}/{(len(titles)-1)//query_batch_size + 1} (大小: {len(batch_titles)})")
            
            # 准备批处理查询
            batch_queries = []
            for title, abstract in zip(batch_titles, batch_abstracts):
                query_text = f"标题：{title}。摘要：{abstract}"
                instructed_query = self.get_detailed_instruct(query_text)
                batch_queries.append(instructed_query)
            
            # 获取查询嵌入
            query_embeddings = self.get_embeddings_batch(batch_queries)
            
            if query_embeddings.numel() == 0:
                all_results.extend([[] for _ in range(len(batch_queries))])
                continue
            
            # 计算相似度
            scores = query_embeddings @ discipline_embeddings.T  # [batch_size, num_disciplines]
            scores = scores.cpu().numpy()
            
            # 为每个查询获取topk学科
            for query_scores in scores:
                top_indices = np.argsort(query_scores)[::-1][:topk]
                results = []
                for idx in top_indices:
                    code = discipline_codes[idx]
                    name = self.code2name.get(code, "")
                    score = float(query_scores[idx])
                    results.append((f"{code} {name}", score))
                all_results.append(results)
            
            # 清理GPU缓存
            if self.device == "cuda":
                torch.cuda.empty_cache()
        
        return all_results

import pandas as pd
import os
from pathlib import Path
import glob
from typing import List, Tuple
import json

def process_csv_files_with_scorer(scorer, input_dir: str, output_dir: str, batch_size: int = 32, overwrite: bool = False):
    """
    批量处理CSV文件中的论文标题和摘要，更新list_title_abs字段
    
    Args:
        scorer: QwenDisciplineScorer实例
        input_dir: 输入CSV文件目录
        output_dir: 输出目录
        batch_size: 批处理大小
        overwrite: 是否覆盖已存在的文件
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有CSV文件
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    print(f"📁 找到 {len(csv_files)} 个CSV文件")
    
    total_processed = 0
    skipped_files = 0
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        output_file = os.path.join(output_dir, filename)
        
        # 检查输出文件是否已存在
        if os.path.exists(output_file) and not overwrite:
            print(f"⏭️  跳过已存在文件: {filename}")
            skipped_files += 1
            continue
            
        print(f"\n🔍 处理文件: {filename}")
        
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file)
            print(f"  读取到 {len(df)} 行数据")
            
            # 检查必要的列是否存在
            required_columns = ['论文标题', 'CR_摘要', 'list_title_abs']
            if not all(col in df.columns for col in required_columns):
                print(f"  ⚠️ 文件缺少必要列，跳过")
                continue
            
            # 准备批量处理数据
            titles = []
            abstracts = []
            valid_indices = []
            
            for idx, row in df.iterrows():
                title = str(row['论文标题']) if pd.notna(row['论文标题']) else ""
                abstract = str(row['CR_摘要']) if pd.notna(row['CR_摘要']) else ""
                
                # 只处理有标题和摘要的数据
                if title and abstract:
                    titles.append(title)
                    abstracts.append(abstract)
                    valid_indices.append(idx)
            
            print(f"  ✅ 有效数据: {len(titles)}/{len(df)}")
            
            if not titles:
                print(f"  ⚠️ 没有有效数据，跳过")
                continue
            
            # 分批处理以避免内存溢出
            all_results = []
            for i in range(0, len(titles), batch_size):
                batch_titles = titles[i:i+batch_size]
                batch_abstracts = abstracts[i:i+batch_size]
                
                print(f"    处理批次 {i//batch_size + 1}/{(len(titles)-1)//batch_size + 1}")
                
                try:
                    batch_results = scorer.score_batch_memory_efficient(
                        batch_titles, 
                        batch_abstracts, 
                        topk=5,
                        query_batch_size=min(8, batch_size)  # 更小的查询批次
                    )
                    all_results.extend(batch_results)
                    
                    # 清理内存
                    if scorer.device == "cuda":
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"    ❌ 批次处理失败: {e}")
                    # 为失败的批次添加空结果
                    all_results.extend([[] for _ in range(len(batch_titles))])
            
            # 更新list_title_abs字段
            updated_count = 0
            for idx, result_idx in enumerate(valid_indices):
                if idx < len(all_results) and all_results[idx]:
                    # 将结果转换为字符串格式
                    result_str = json.dumps(all_results[idx], ensure_ascii=False)
                    df.at[result_idx, 'list_title_abs'] = result_str
                    updated_count += 1
            
            # 保存更新后的文件
            df.to_csv(output_file, index=False, encoding='utf-8')
            
            print(f"  ✅ 更新完成: {updated_count}/{len(valid_indices)} 行")
            print(f"  💾 文件已保存: {output_file}")
            total_processed += updated_count
            
        except Exception as e:
            print(f"  ❌ 文件处理失败: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n📊 处理统计:")
    print(f"  ✅ 成功处理: {len(csv_files) - skipped_files} 个文件")
    print(f"  ⏭️  跳过文件: {skipped_files} 个")
    print(f"  📝 总共更新: {total_processed} 行数据")

def process_single_csv_with_progress(scorer, csv_file: str, output_file: str, batch_size: int = 16, overwrite: bool = False):
    """
    处理单个CSV文件并显示进度
    
    Args:
        scorer: QwenDisciplineScorer实例
        csv_file: 输入CSV文件路径
        output_file: 输出文件路径
        batch_size: 批处理大小
        overwrite: 是否覆盖已存在的文件
    """
    
    # 检查输出文件是否已存在
    if os.path.exists(output_file) and not overwrite:
        print(f"⏭️  跳过已存在文件: {os.path.basename(output_file)}")
        return
    
    print(f"🔍 处理文件: {csv_file}")
    
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_file)
        print(f"📊 数据量: {len(df)} 行")
        
        # 检查必要的列
        if '论文标题' not in df.columns or 'CR_摘要' not in df.columns:
            print("❌ 文件缺少'论文标题'或'CR_摘要'列")
            return
        
        # 准备数据
        titles = []
        abstracts = []
        valid_indices = []
        
        for idx, row in df.iterrows():
            title = str(row['论文标题']) if pd.notna(row['论文标题']) else ""
            abstract = str(row['CR_摘要']) if pd.notna(row['CR_摘要']) else ""
            
            if title and abstract and len(abstract) > 10:  # 摘要至少10个字符
                titles.append(title)
                abstracts.append(abstract)
                valid_indices.append(idx)
        
        print(f"✅ 有效数据: {len(titles)}/{len(df)}")
        
        if not titles:
            print("⚠️ 没有有效数据")
            return
        
        # 处理数据
        print("🚀 开始计算学科分数...")
        all_results = scorer.score_batch_memory_efficient(
            titles, abstracts, topk=5, query_batch_size=min(8, batch_size)
        )
        
        # 更新数据框
        for idx, result_idx in enumerate(valid_indices):
            if idx < len(all_results):
                df.at[result_idx, 'list_title_abs'] = json.dumps(all_results[idx], ensure_ascii=False)
        
        # 保存结果
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"💾 结果已保存到: {output_file}")
        
        # 显示一些统计信息
        successful_updates = sum(1 for idx in valid_indices if idx < len(all_results) and all_results[idx])
        print(f"📈 成功更新: {successful_updates}/{len(valid_indices)}")
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()

def run_discipline_scoring_pipeline(overwrite: bool = False):
    """
    运行学科评分流水线
    
    Args:
        overwrite: 是否覆盖已存在的输出文件
    """
    try:
        # 初始化评分器
        print("🚀 初始化QwenDisciplineScorer...")
        scorer = QwenDisciplineScorer(
            use_flash_attention=False,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # 设置路径
        input_dir = "./data/04input_data"  # 根据您的实际路径调整
        output_dir = "./data/05processed_data"
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        print("📁 开始批量处理CSV文件...")
        if overwrite:
            print("⚠️  覆盖模式: 将覆盖已存在的输出文件")
        else:
            print("🔒 跳过模式: 将跳过已存在的输出文件")
        
        # 方法1: 批量处理所有文件
        process_csv_files_with_scorer(
            scorer=scorer,
            input_dir=input_dir,
            output_dir=output_dir,
            batch_size=16,  # 保守的批大小
            overwrite=overwrite
        )
        
        # 方法2: 或者逐个文件处理（取消注释使用）
        # csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
        # for csv_file in csv_files[:1]:  # 只处理第一个文件作为测试
        #     filename = os.path.basename(csv_file)
        #     output_file = os.path.join(output_dir, filename)
        #     process_single_csv_with_progress(scorer, csv_file, output_file, overwrite=overwrite)
        
        print("🎉 学科评分流水线完成!")
        
    except Exception as e:
        print(f"❌ 流水线执行失败: {e}")
        import traceback
        traceback.print_exc()

# 新增函数：检查处理进度
def check_processing_progress(input_dir: str, output_dir: str):
    """
    检查处理进度，显示哪些文件已处理，哪些未处理
    """
    input_files = set([os.path.basename(f) for f in glob.glob(os.path.join(input_dir, "*.csv"))])
    output_files = set([os.path.basename(f) for f in glob.glob(os.path.join(output_dir, "*.csv"))])
    
    processed = input_files & output_files
    unprocessed = input_files - output_files
    
    print(f"\n📊 处理进度检查:")
    print(f"📁 输入目录: {input_dir}")
    print(f"📁 输出目录: {output_dir}")
    print(f"✅ 已处理文件: {len(processed)} 个")
    print(f"⏳ 未处理文件: {len(unprocessed)} 个")
    
    if unprocessed:
        print("\n📋 未处理文件列表:")
        for file in sorted(unprocessed):
            print(f"  - {file}")

# 测试函数
def test_small_batch():
    """
    测试小批量处理
    """
    try:
        print("🧪 测试小批量处理...")
        scorer = QwenDisciplineScorer(
            use_flash_attention=False,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # 创建测试数据
        test_data = [
            {
                "论文标题": "深度学习在自然语言处理中的应用",
                "CR_摘要": "本文研究了深度学习技术在自然语言处理领域的各种应用，包括文本分类、机器翻译和情感分析等。"
            },
            {
                "论文标题": "经济学中的市场分析",
                "CR_摘要": "本文通过实证分析研究了市场经济中的供需关系和价格形成机制。"
            }
        ]
        
        titles = [item["论文标题"] for item in test_data]
        abstracts = [item["CR_摘要"] for item in test_data]
        
        results = scorer.score_batch_memory_efficient(titles, abstracts, topk=3)
        
        print("测试结果:")
        for i, (title, result) in enumerate(zip(titles, results)):
            print(f"论文 {i+1}: {title}")
            for discipline, score in result:
                print(f"  {discipline}: {score:.4f}")
            print()
            
    except Exception as e:
        print(f"测试失败: {e}")

if __name__ == "__main__":
    # 检查处理进度
    input_dir = "./data/04input_data"
    output_dir = "./data/05processed_data"
    check_processing_progress(input_dir, output_dir)
    
    # 运行测试
    print("\n🧪 运行小批量测试...")
    test_small_batch()
    
    # 运行完整流水线（默认不覆盖已存在文件）
    print("\n🚀 运行完整流水线...")
    run_discipline_scoring_pipeline(overwrite=True)
    
    # 如果需要覆盖已存在文件，使用：
    # run_discipline_scoring_pipeline(overwrite=True)
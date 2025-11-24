# -*- coding: utf-8 -*-
# Modified for subjectcross project structure
# Created by Messi & ChatGPT (2025/12)

import argparse
from loguru import logger
import os
import sys
from pathlib import Path

# ============================================================
# 自动推断项目根目录（非常重要）
# ============================================================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]       # subjectcross/
FINETUNE_DIR = PROJECT_ROOT / "finetune"
SCRIPT_DIR = FINETUNE_DIR / "script"

# 让 component/ 可以被 import
sys.path.append(str(FINETUNE_DIR))
sys.path.append(str(SCRIPT_DIR))

# 将工作目录切换到项目根
os.chdir(PROJECT_ROOT)
print(f"[INFO] 当前工作目录切换为: {PROJECT_ROOT}")

# ============================================================
# 环境变量
# ============================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ============================================================
# 导入组件
# ============================================================
from os.path import join
import random
from component.collator import PretrainCollator, SFTDataCollator
from component.argument import CustomizedArguments
from component.template import template_dict
from component.load_peft import load_model, load_tokenizer
from component.dataset import (
    UnifiedSFTDataset,
    ChatGLM2SFTDataset,
    ChatGLM3SFTDataset,
)
from transformers import (
    set_seed,
    HfArgumentParser,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset, concatenate_datasets
import datasets
from itertools import chain
from tqdm import tqdm
import json
from torch import nn
import torch

# ============================================================
# Step 1 —— 读取 args.json
# ============================================================
def setup_everything():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_args_file",
        type=str,
        default=str(SCRIPT_DIR / "args.json")
    )
    parser.add_argument("--local_rank", type=int, help="")
    args = parser.parse_args()

    train_args_file = args.train_args_file
    print(f"[INFO] 加载训练配置文件: {train_args_file}")

    # 加载训练参数
    parser = HfArgumentParser((CustomizedArguments, TrainingArguments))
    args, training_args = parser.parse_json_file(json_file=train_args_file)

    # 创建输出目录
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    logger.add(join(training_args.output_dir, 'train.log'))
    logger.info(f"train_args: {training_args}")

    # 保存完整配置
    with open(train_args_file, "r") as f:
        config_json = json.load(f)
    with open(join(training_args.output_dir, 'train_args.json'), "w") as f:
        json.dump(config_json, f, indent=4)

    # 设置随机种子
    set_seed(training_args.seed)

    return args, training_args


# ============================================================
# SFT 数据加载
# ============================================================
def load_sft_dataset(args, tokenizer):
    train_file = f"{args.dataset}/train.jsonl"
    template = template_dict[args.template_name]

    logger.info(f"[INFO] 加载 SFT 数据集: {train_file}")

    if 'chatglm2' in args.model_name_or_path.lower():
        return ChatGLM2SFTDataset(train_file, tokenizer, args.max_seq_length, template)
    elif 'chatglm3' in args.model_name_or_path.lower():
        return ChatGLM3SFTDataset(train_file, tokenizer, args.max_seq_length, template)
    else:
        return UnifiedSFTDataset(train_file, tokenizer, args.max_seq_length, template)


# ============================================================
# Step 2 —— 初始化所有组件
# ============================================================
def init_components(args, training_args):
    training_args.ddp_find_unused_parameters = False
    logger.info('Initializing components...')

    # tokenizer
    tokenizer = load_tokenizer(args)

    # Dataset
    logger.info("Train model with SFT task")
    args.train_dataset = load_sft_dataset(args, tokenizer)
    random.shuffle(args.train_dataset.data_list)

    # Collator
    data_collator = SFTDataCollator(tokenizer, args.max_seq_length)

    # Model
    model = load_model(args, training_args)

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=args.train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    return trainer


# ============================================================
# 主流程
# ============================================================
def main():
    args, training_args = setup_everything()

    trainer = init_components(args, training_args)

    logger.info("*** Starting training ***")
    train_result = trainer.train()

    # 保存权重
    final_save_path = join(training_args.output_dir)
    trainer.save_model(final_save_path)
    print(f"模型已保存到: {final_save_path}")

    # 显存
    print(f'最大显存占用：{round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2)} GB')

    # 指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()

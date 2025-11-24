import argparse
from loguru import logger
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
    Trainer,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset, concatenate_datasets
import datasets
from itertools import chain
from tqdm import tqdm
import json
from torch import nn
import torch
from script.evaluation.evaluate_scierc import evaluate_scierc
from script.evaluation.evaluate_scicite import evaluate_scicite
from script.evaluation.evaluate_scinli import evaluate_scinli

eval_dict = {
    "scierc": evaluate_scierc,
    "scicite": evaluate_scicite,
    "scinli": evaluate_scinli
    }


def setup_everything(train_args_file):
    parser = argparse.ArgumentParser()
    # parser.add_argument("--train_args_file", type=str, default='train_args/pretrain/full/bloom-1b1-pretrain-full.json', help="")
    # parser.add_argument("--train_args_file", type=str, default='train_args/sft/prefix/llama2-7b-sft-prefix.json', help="")
    parser.add_argument("--local_rank", type=int, help="")
    args = parser.parse_args()
    # train_args_file = args.train_args_file
    # 读取训练的参数配置
    parser = HfArgumentParser((CustomizedArguments, TrainingArguments))
    # 解析得到自定义参数，以及自带参数
    args, training_args = parser.parse_json_file(json_file=train_args_file)
    # 创建输出目录
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    logger.add(join(training_args.output_dir, 'train.log'))
    logger.info("train_args:{}".format(training_args))
    # 加载训练配置文件
    with open(train_args_file, "r") as f:
        train_args = json.load(f)
    # 保存训练参数到输出目录
    with open(join(training_args.output_dir, 'train_args.json'), "w") as f:
        json.dump(train_args, f, indent=4)
    # 设置随机种子
    set_seed(training_args.seed)
    return args, training_args


def load_pretrain_dataset(training_args, args, tokenizer):
    """
    多线程预处理预训练数据
    """
    def tokenize_function(examples):
        output = tokenizer(examples["text"])
        output = {'input_ids': output.input_ids}
        return output

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= max_seq_length:
            total_length = (total_length // max_seq_length) * max_seq_length
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    data_path = args.train_file
    max_seq_length = args.max_seq_length
    # 创建缓存路径
    cache_dir = join(data_path, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    logger.info('Pretraining data path: {}'.format(data_path))

    # 扫描所有jsonl文件
    logger.info('Scanning all the training file...')
    files = []
    for root, dir_names, file_names in os.walk(data_path):
        for file_name in file_names:
            file = join(root, file_name)
            if file_name.endswith('.jsonl'):
                files.append(file)
    logger.info(f'Total num of training file: {len(files)}')

    # 预处理所有文本，将其id化，并且进行packing操作
    with training_args.main_process_first(desc="dataset map tokenization and grouping"):
        pretrain_dataset = []  # 汇总所有dataset
        for idx, file in enumerate(tqdm(files)):
            logger.info(f'Loading file: {file}')
            file_name = os.path.basename(file)
            file_name = file_name.replace('.jsonl', '')
            cache_path = os.path.join(cache_dir, file_name)
            os.makedirs(cache_path, exist_ok=True)

            try:
                processed_dataset = datasets.load_from_disk(cache_path, keep_in_memory=False)
                logger.info(f'Finished loading datasets-{file_name} from cache')
            except Exception:
                tmp_cache_path = join(cache_path, 'tmp')    # 临时缓存目录，会被自动删除
                logger.info(f'There is no cache of file {file_name}, start preprocessing...')
                raw_dataset = load_dataset("json", data_files=file, cache_dir=tmp_cache_path, keep_in_memory=False)
                tokenized_dataset = raw_dataset.map(
                    tokenize_function,
                    batched=True,
                    num_proc=args.tokenize_num_workers,
                    remove_columns="text",
                    load_from_cache_file=True,
                    keep_in_memory=False,
                    cache_file_names={k: os.path.join(tmp_cache_path, 'tokenized.arrow') for k in raw_dataset},
                    desc="Running tokenizer on dataset",
                )
                grouped_datasets = tokenized_dataset.map(
                    group_texts,
                    batched=True,
                    num_proc=args.tokenize_num_workers,
                    load_from_cache_file=True,
                    keep_in_memory=False,
                    cache_file_names={k: os.path.join(tmp_cache_path, 'grouped.arrow') for k in tokenized_dataset},
                    desc=f"Grouping texts in chunks of {max_seq_length}",
                )
                processed_dataset = grouped_datasets
                processed_dataset.save_to_disk(cache_path)
                # 删除临时目录
                # shutil.rmtree(tmp_cache_path)

            logger.info(f"Training number of {file_name}: {len(processed_dataset['train'])}")
            if idx == 0:
                pretrain_dataset = processed_dataset['train']
            else:
                assert pretrain_dataset.features.type == processed_dataset["train"].features.type
                pretrain_dataset = concatenate_datasets([pretrain_dataset, processed_dataset["train"]])
    logger.info(f"Total training number: {len(pretrain_dataset)}")
    return pretrain_dataset


def load_sft_dataset(args, tokenizer):
    
    test_dataset = None
    train_file = f'{args.dataset}/train.jsonl'
    template = template_dict[args.template_name]
    if 'chatglm2' in args.model_name_or_path.lower():
        logger.info('Loading data with ChatGLM2SFTDataset')
        train_dataset = ChatGLM2SFTDataset(train_file, tokenizer, args.max_seq_length, template)
    elif 'chatglm3' in args.model_name_or_path.lower():
        logger.info('Loading data with ChatGLM3SFTDataset')
        train_dataset = ChatGLM3SFTDataset(train_file, tokenizer, args.max_seq_length, template)
    else:
        logger.info('Loading data with UnifiedSFTDataset')
        train_dataset = UnifiedSFTDataset(train_file, tokenizer, args.max_seq_length, template)
        if args.do_prediction is not None:
            test_file = f'{args.dataset}/test.jsonl'
            test_dataset = UnifiedSFTDataset(test_file, tokenizer, args.max_seq_length, template)
            
    return train_dataset, test_dataset

def _prepare_model_for_training(model: nn.Module):
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)


def init_components(args, training_args):
    """
    初始化各个组件
    """
    training_args.ddp_find_unused_parameters = False
    logger.info('Initializing components...')

    # 加载tokenizer
    tokenizer = load_tokenizer(args)

    # 初始化dataset和collator
    if args.task_type == 'pretrain':
        logger.info('Train model with pretrain task')
        args.train_dataset = load_pretrain_dataset(training_args, args, tokenizer)
        data_collator = PretrainCollator(tokenizer, args.max_seq_length)
    else:
        logger.info('Train model with sft task')
        args.train_dataset, args.test_dataset = load_sft_dataset(args, tokenizer)
        data_collator = SFTDataCollator(tokenizer, args.max_seq_length)

    # 加载model
    model = load_model(args, training_args)

    return data_collator, model, tokenizer


def save_and_predict(args, training_args, model, tokenizer, test_dataloader, iter):
    # saving model
    peft_model_id = f"{training_args.output_dir}/{training_args.save_strategy}-{iter+1}"
    model.save_pretrained(peft_model_id)
    model.eval()
    model_name = args.model_name_or_path.split('/')[-1]
    output_data_path = f'output/{model_name}/{args.dataset}/test-{args.train_mode}-{training_args.save_strategy}-{iter+1}-{training_args.learning_rate}.jsonl'
    # if os.path.exists(output_data_path):
    #     continue
    f = open(output_data_path, "w", encoding="utf-8")
    template = template_dict[args.template_name]
    for step, batch in enumerate(tqdm(test_dataloader)):
        test_dict = json.loads(test_dataloader.dataset.data_list[step])
        batch = {k: v.to(model.base_model.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model.generate(input_ids=batch['input_ids'], do_sample=False, max_new_tokens=128)
        # output_str = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0].split('[/INST]')[-1].strip()
        outputs = outputs.tolist()[0][len(batch['input_ids'][0]):]
        response = tokenizer.decode(outputs)
        response = response.strip().replace(template.stop_word, "").strip()
        test_dict['conversation'][0]['pred'] = response
        f.write(json.dumps(test_dict, ensure_ascii=False) + "\n")
        f.flush()  # flush the buffer to disk
        # if step % 10 == 0:
            # print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0])
            # print(test_dict)
    f.close()
    eval_dict[args.dataset](output_data_path)

def main(train_args_file):
    # 进行一些配置和检查
    args, training_args = setup_everything(train_args_file)
    # 加载各种组件
    data_collator, model, tokenizer = init_components(args, training_args)
    train_dataloader = torch.utils.data.DataLoader(args.train_dataset, collate_fn=data_collator, batch_size=training_args.per_device_train_batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(args.test_dataset, collate_fn=data_collator, batch_size=training_args.per_device_eval_batch_size)
    # model
    # optimizer and lr scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=(len(train_dataloader) * training_args.num_train_epochs),
    )
    # 开始训练
    logger.info("*** starting training ***")
    
    # training and evaluation
    total_step = 0
    for epoch in range(training_args.num_train_epochs):
        model.train()
        total_loss = 0
        
        for step, batch in enumerate(tqdm(train_dataloader)):
            total_step = total_step + 1
            batch = {k: v.to(model.base_model.device) for k, v in batch.items()}
            #         print(batch)
            #         print(batch["input_ids"].shape)
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            loss = loss / training_args.gradient_accumulation_steps
            loss.backward()
            if (step+1) % training_args.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            if args.do_prediction and training_args.save_strategy == "steps" and (total_step+1) % training_args.save_steps == 0:
                # 保存模型并推理
                save_and_predict(args, training_args, model, tokenizer, test_dataloader, total_step)
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        print(f"{epoch=}: {train_ppl=} {train_epoch_loss=}")
        print(f'最大显存使用量：{round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2)} G')
        if args.do_prediction and training_args.save_strategy == "epoch":
            # 保存模型并推理
            save_and_predict(args, training_args, model, tokenizer, test_dataloader, epoch)


if __name__ == "__main__":
    train_args_file = 'script/args.json'
    main(train_args_file)

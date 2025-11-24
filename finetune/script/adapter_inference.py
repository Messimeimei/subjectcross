import os
import sys
sys.path.append("/data01/public/yifan/grant_match_v3")
os.chdir("/data01/public/yifan/grant_match_v3")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time
from transformers import AutoTokenizer, AutoConfig, AddedToken
import torch
from loguru import logger
from tqdm import tqdm
import json
import sys
sys.path.append(os.getcwd())
from component.utils import ModelUtils
from component.template import template_dict


def build_prompt_chatglm3(tokenizer, query, history, system=None):
    history.append({"role": 'user', 'message': query})
    # system
    input_ids = tokenizer.get_prefix_tokens() + \
                [tokenizer.get_command(f"<|system|>")] + \
                tokenizer.encode(system, add_special_tokens=False)
    # convs
    for item in history:
        role, message = item['role'], item['message']
        if role == 'user':
            tokens = [tokenizer.get_command(f"<|user|>")] + \
                     tokenizer.encode(message, add_special_tokens=False) + \
                     [tokenizer.get_command(f"<|assistant|>")]
        else:
            tokens = tokenizer.encode(message, add_special_tokens=False) + [tokenizer.eos_token_id]
        input_ids += tokens

    return input_ids


def build_prompt(tokenizer, template, query, history, system=None):
    template_name = template.template_name
    system_format = template.system_format
    user_format = template.user_format
    assistant_format = template.assistant_format
    system = system if system is not None else template.system

    if template_name == 'chatglm2':
        prompt = tokenizer.build_prompt(query, history)
        input_ids = tokenizer.encode(prompt)
    elif template_name == 'chatglm3':
        input_ids = build_prompt_chatglm3(tokenizer, query, history, system)
    else:
        history.append({"role": 'user', 'message': query})
        input_ids = []

        # setting system information
        if system_format is not None:
            # system信息不为空
            if system is not None:
                system_text = system_format.format(content=system)
                input_ids = tokenizer.encode(system_text, add_special_tokens=False)
        # concat conversation
        for item in history:
            role, message = item['role'], item['message']
            if role == 'user':
                message = user_format.format(content=message, stop_token=tokenizer.eos_token)
            else:
                message = assistant_format.format(content=message, stop_token=tokenizer.eos_token)
            tokens = tokenizer.encode(message, add_special_tokens=False)
            input_ids += tokens
            # input_ids = tokenizer(system_text + message, return_tensors="pt")
    input_ids = torch.tensor([input_ids], dtype=torch.long)

    return input_ids


def load_tokenizer(model_name_or_path):
    # config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    # 加载tokenzier
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        # model_max_length=497,
        use_fast=False
        # llama不支持fast
        # use_fast=False if config.model_type == 'llama' else True
    )

    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # assert tokenizer.pad_token_id is not None, "pad_token_id should not be None"
    return tokenizer


def main(
        model_name_or_path: str, 
        adapter_name_or_path: str, 
        data_path: str, 
        output_path: str,
        adapter_type: str,
        template_name: str, 
    ):
    
    # 是否使用4bit进行推理，能够节省很多显存，但效果可能会有一定的下降
    load_in_4bit = True

    template = template_dict[template_name]
    
    # 生成超参配置
    max_new_tokens = 2048 # yifan modify
    top_p = 0.9
    temperature = 0.35
    repetition_penalty = 1.0

    # 加载模型
    logger.info(f'Loading model from: {model_name_or_path}')
    logger.info(f'adapter_name_or_path: {adapter_name_or_path}')
    model = ModelUtils.load_model(
        model_name_or_path,
        load_in_4bit=load_in_4bit,
        adapter_name_or_path=adapter_name_or_path
    ).eval()
    # tokenizer = load_tokenizer(model_name_or_path if adapter_name_or_path is None else adapter_name_or_path)
    tokenizer = load_tokenizer(model_name_or_path)

    if template_name == 'chatglm2':
        stop_token_id = tokenizer.eos_token_id
    elif template_name == 'chatglm3':
        stop_token_id = [tokenizer.eos_token_id, tokenizer.get_command("<|user|>"), tokenizer.get_command("<|observation|>")]
    else:
        if template.stop_word is None:
            template.stop_word = tokenizer.eos_token
        stop_token_id = tokenizer.encode(template.stop_word, add_special_tokens=False)
        assert len(stop_token_id) >= 1  # yifan 0724 add
        # assert len(stop_token_id) == 1  # yifan 0724 delete
        stop_token_id = stop_token_id[0]

    # 读取数据集
    with open(data_path, 'r', encoding='utf8') as f:
        data_list = f.readlines()

    # 写入结果
    ## 创建output文件夹
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    ## 创建并写入jsonl文件
    f = open(output_path, "w", encoding="utf-8")
    for data in tqdm(data_list):
        data = eval(data)
        # 若adapter_type中包含prompt、prefix、ptuning
        if any(adapter in adapter_type for adapter in ["prompt", "prefix", "ptuning"]):
            query = data['conversation'][0]['human'].split('Sentence:\n')[1]
        else:
            query = data['conversation'][0]['human']
        query = query.strip()
        # print(query+'\n')
        input_ids = build_prompt(tokenizer, template, query, history=[], system=None).to(model.device)
        # Without streaming
        with torch.no_grad():
            outputs = model.generate(
                    input_ids=input_ids, 
                    # attention_mask=input_ids["attention_mask"],
                    max_new_tokens=max_new_tokens, 
                    # use_cache=False,
                    do_sample=False,
                    # do_sample=True, 
                    # top_p=top_p, temperature=temperature, repetition_penalty=repetition_penalty, 
                    eos_token_id=stop_token_id
            )
        outputs = outputs.tolist()[0][len(input_ids[0]):]
        response = tokenizer.decode(outputs)
        response = response.strip().replace(template.stop_word, "").strip()
        # print(response)
        # update history
        data['conversation'][0]['pred'] = response
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
        f.flush()  # flush the buffer to disk

    f.close()
    time.sleep(5)
    
    # eval_dict[dataset](output_path)

if __name__ == '__main__':
    # [ 微调后inference ]

    # 结构化摘要抽取的推理
    # i = 5
    # model_name  = 'llama2-7b'
    # adapter_type = 'qlora'
    # template_name = 'llama2'
    # ckpt = 1800
    # model_name_or_path =  "/data01/public/yifan/model/Llama-2-7b-chat-hf"
    # adapter_name_or_path = f"/data01/public/yifan/grant_match/output/data0804/llama-7b-simple/checkpoint-{ckpt}"
    # data_path = f"/data01/public/yifan/grant_match/data/program_group/temp_element/program_group_element_{i}.jsonl"
    # output_path = f"/data01/public/yifan/grant_match/data/program_group/temp_element/predict_program_group_element_{i}.jsonl"

    # 匹配度的推理
    # i = 7
    # # year = 2015
    # model_name  = 'llama2-7b'
    # adapter_type = 'qlora'
    # template_name = 'llama2'
    # ckpt = 3500
    # model_name_or_path =  "/data01/public/yifan/model/Llama-2-7b-chat-hf"
    # adapter_name_or_path = f"/data01/public/yifan/grant_match/output/data_temp/llama-7b-0923-random-content-all/checkpoint-{ckpt}"
    # data_path = f"/data01/public/yifan/grant_match/data/program_group/temp_match/program_group_match_{i}.jsonl"
    # output_path = f"/data01/public/yifan/grant_match/data/program_group/temp_match/predict_program_group_match_{i}.jsonl"

    # [mistral]
    # model_name  = 'mistral-7b'  # Llama-2-7b-chat-hf/Llama-2-13b-chat-hf/chatglm3-6b/Mistral-7B-Instruct-v0.1/Mixtral-8x7B-Instruct-v0.1
    # adapter_type = 'qlora'  # qlora/lora/ptuning/ptuning-v2/prefix/prompt
    # template_name = 'mistral' # llama2/chatglm3/mistral/mixtral
    # ckpt = 2400 # 1800
    
    # model_output_dir = "/home/llmtrainer/yf_project/research_question_tree_jy/output/data0923/mistral-7b"
    # model_name_or_path =  "/data_share/model_hub/Mixtral/Mistral-7B-Instruct-v0.3"
    # dataset_dir = "/home/llmtrainer/yf_project/research_question_tree_jy/data/data0923/random_content"

    # [llama]
    # model_name  = 'llama2-7b'  # Llama-2-7b-chat-hf/Llama-2-13b-chat-hf/chatglm3-6b/Mistral-7B-Instruct-v0.1/Mixtral-8x7B-Instruct-v0.1
    # adapter_type = 'qlora'  # qlora/lora/ptuning/ptuning-v2/prefix/prompt
    # template_name = 'llama2' # llama2/chatglm3/mistral/mixtral
    # ckpt = 3500 # 1800
    
    # model_output_dir = "/home/llmtrainer/yf_project/research_question_tree_jy/output/data_temp/llama-7b-0923-random-content-all"
    # model_name_or_path =  "/data_share/model_hub/llama/Llama-2-7b-chat-hf"
    # dataset_dir = "/home/llmtrainer/yf_project/research_question_tree_jy/data/data0923/random_content_all"
    
    # adapter_name_or_path = model_output_dir + f"/checkpoint-{ckpt}"
    # data_path = dataset_dir + "/test.jsonl"
    # output_path = dataset_dir + f"/test_predict_{model_name}_{ckpt}.jsonl"


    # 停顿时间
    # stop_hour = 0.1
    # logger.info(f'{stop_hour}小时后开始运行') 
    # for temp in tqdm([1 for i in range(60)]):
    #     time.sleep(stop_hour * 60)
    
    # model_name = "qwen3-8b"
    # model_name_or_path =  "/data01/public/yifan/model/Qwen3-8B"

    # adapter_name_or_path = f"/data01/public/yifan/grant_match_v3/output/data202505/{model_name}-select/checkpoint-3600"
    # data_path = "/data01/public/yifan/grant_match/data/data202505/select/finetune_test.jsonl"
    # output_path = f"/data01/public/yifan/grant_match/data/data202505/select/finetune_{model_name}_test.jsonl"
    # adapter_type = "qlora"
    # template_name = "qwen3"

    # 不同数据量
    adapter_type = "qlora"
    template_name = "qwen3"
    model_name = "qwen3-8b"
    model_name_or_path =  "/data01/public/yifan/model/Qwen3-8B"

    # for checkpoint in range(1000, 11000, 1000):
    #     adapter_name_or_path = f"/data01/public/yifan/grant_match_v3/output/match/data251019/{model_name}/checkpoint-{checkpoint}"
    #     # data_path = "/data01/public/yifan/grant_match_v3/data_match/1019/test.jsonl"
    #     # output_path = f"/data01/public/yifan/grant_match_v3/data_match/1019/{model_name}_{checkpoint}_test.jsonl"
    #     # main(model_name_or_path, adapter_name_or_path, data_path, output_path, adapter_type, template_name)
    #     data_path = "/data01/public/yifan/grant_match_v3/data_match/1019/human_test.jsonl"
    #     output_path = f"/data01/public/yifan/grant_match_v3/data_match/1019/{model_name}_{checkpoint}_human_test.jsonl"
    #     main(model_name_or_path, adapter_name_or_path, data_path, output_path, adapter_type, template_name)

    # 主函数
    for checkpoint in range(5000, 500, -1000):
        adapter_name_or_path = f"/data01/public/yifan/grant_match_v3/output/match/epoch10/qwen3-8b/checkpoint-{checkpoint}"
        data_path = "/data01/public/yifan/grant_match_v3/data_match/1019/human_test.jsonl"
        output_path = f"/data01/public/yifan/grant_match_v3/data_match/1019/{model_name}_{checkpoint}_human_test_epoch10.jsonl"
        main(model_name_or_path, adapter_name_or_path, data_path, output_path, adapter_type, template_name)

    # for checkpoint in range(1400, 0, -100):
    #     adapter_name_or_path = f"/data01/public/yifan/grant_match_v3/output/match/data200/qwen3-8b/checkpoint-{checkpoint}" # 
    #     data_path = "/data01/public/yifan/grant_match_v3/data_match/1019/human_test.jsonl"
    #     output_path = f"/data01/public/yifan/grant_match_v3/data_match/1019/{model_name}_{checkpoint}_human_test_data200.jsonl"
    #     main(model_name_or_path, adapter_name_or_path, data_path, output_path, adapter_type, template_name)
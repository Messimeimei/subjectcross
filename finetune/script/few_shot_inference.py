# from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import sys
import fire
import torch
import json
from tqdm import tqdm
import time
from transformers import LlamaTokenizer, AutoTokenizer, set_seed
from utils import load_model
import pandas as pd
from loguru import logger

SYS_MESSAGE = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."

def main(model_path, data_path, output_path):
    # 保留原有参数
    quantization=False
    max_new_tokens=512 #The maximum numbers of tokens to generate
    seed=42 #seed value for reproducibility
    do_sample=True #Whether or not to use sampling ; use greedy decoding otherwise.
    min_length=None #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache=True  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p=1.0 # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature=1 # [optional] The value used to modulate the next token probabilities.
    top_k=50 # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty=1.0 #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty=1 #[optional] Exponential penalty to the length that is used with beam-based generation.
    
    # 读取数据集
    with open(data_path, 'r', encoding='utf8') as f:
        data_list = f.readlines()

    # 模型设置
    set_seed(seed)
    model = load_model(model_path, quantization).eval()
    print("load 完成")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 写入结果
    ## 创建output文件夹
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    ## 创建并写入jsonl文件
    f = open(output_path, "w", encoding="utf-8")
    for data in tqdm(data_list):
        data = eval(data)
        # 设置message
        message = []
        if "llama2" in model_path:
            message.append({"role": "system", "content": f"{SYS_MESSAGE}"})
        message.append({"role": "user", "content": data["conversation"][0]["human"]})

        prompt = tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )

        batch = tokenizer(prompt, return_tensors="pt")
        batch = {k: v.to("cuda") for k, v in batch.items()}
        # start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **batch,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                min_length=min_length,
                use_cache=use_cache,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
            )
        a = int(batch['attention_mask'].shape[1])

        for res_el in outputs.tolist():
            answer = tokenizer.decode(res_el[a:], skip_special_tokens=True).strip()
            # print(answer)

        data['conversation'][0]['pred'] = answer
        # 数据存储
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
        f.flush()  # flush the buffer to disk

    f.close()
    time.sleep(5)


# MODEL_DICT = {
    # "llama2_7b_chat": "/data01/share/model_hub/Llama-2-7b-chat-hf",
    # "llama2_13b_chat": "/data_share/model_hub/llama/Llama-2-13b-chat-hf",
    # "llama2_70b_chat": "/data_share/model_hub/llama/Llama-2-70b-chat-hf",
    # "llama3_8b_chat": "/data_share/model_hub/llama/Meta-Llama-3.1-8B-Instruct",
    # "llama3_70b_chat": "/data_share/model_hub/llama/Meta-Llama-3.1-70B-Instruct",
    # "mistral-7b-v0.1": "/data_share/model_hub/Mixtral/Mistral-7B-Instruct-v0.1",
    # "mixtral_8x7b-v0.1": "/data_share/model_hub/Mixtral/Mixtral-8x7B-Instruct-v0.1",
    # "qwen_14b_chat": "/data_share/model_hub/qwen/Qwen-14B-Chat",
    # "qwen_72b_chat": "/data_share/model_hub/qwen/Qwen-72B-Chat",
# }

MODEL_DICT = {
    "Llama-2-7B": "/data01/share/model_hub/Llama-2-7b-chat-hf",
    "Llama-2-13B": "/data01/share/model_hub/Llama-2-13b-chat-hf",
    "Llama-2-70B": "/data01/share/model_hub/Meta-Llama-3.1-70B-Instruct",
    "Llama-3-8B": "/data01/share/model_hub/Meta-Llama-3.1-8B-Instruct",
    "Llama-3-70B": "/data01/share/model_hub/Llama-2-70b-chat-hf",
    "Mistral-7B": "/data01/public/yifan/model/Mistral-7B-Instruct-v0.3",
    "Qwen-3-0.6B": "/data01/public/yifan/model/Qwen3-0.6B",
    "Qwen-3-4B": "/data01/public/yifan/model/Qwen3-4B", 
    "Qwen-3-8B": "/data01/public/yifan/model/Qwen3-8B",
    "Qwen-3-32B": "/data01/public/yifan/model/Qwen3-32B"
}

if __name__ == "__main__":
    model_name = "Qwen-3-32B"
    # method = "one"
    model_path = MODEL_DICT[model_name]
    data_path = f"/data01/public/yifan/grant_match_v3/data_match/0721/human_test.jsonl"
    output_path = f"/data01/public/yifan/grant_match_v3/data_match/0721/{model_name}_human_test.jsonl"
    try:
        main(model_path, data_path, output_path)
    except Exception as e:
        logger.info(e)
        logger.info(f'\n\n\n===== error =====\nmodel name: {model_name}\nmethod: {method} shot\noutput path: {output_path}\n=================\n\n\n')
    else:
        logger.info(f'\n\n\n===== finish =====\nmodel name: {model_name}\nmethod: {method} shot\noutput path: {output_path}\n==================\n\n\n')

    # for model_name in ["mixtral_8x7b-v0.1"]: # "mixtral_8x7b-v0.1", "mistral-7b-v0.1", "llama3_8b_chat"
    #     for method in ["zero", "one"]:
    #         model_path = MODEL_DICT[model_name]
    #         data_path = f"/home/llmtrainer/yf_project/research_question_tree_jy/data/dataset0804/test_for_paper_data/{method}_shot_test_40.jsonl"
    #         output_path = f"/home/llmtrainer/yf_project/research_question_tree_jy/data/dataset0804/test_for_paper_data/{method}_shot_test_40_predict_{model_name}.jsonl"
    #         logger.info(f'\n\n\n===== start =====\nmodel name: {model_name}\nmethod: {method} shot\noutput path: {output_path}\n=================\n\n\n')
    #         try:
    #             main(model_path, data_path, output_path)
    #         except Exception as e:
    #             logger.info(e)
    #             logger.info(f'\n\n\n===== error =====\nmodel name: {model_name}\nmethod: {method} shot\noutput path: {output_path}\n=================\n\n\n')
    #         else:
    #             logger.info(f'\n\n\n===== finish =====\nmodel name: {model_name}\nmethod: {method} shot\noutput path: {output_path}\n==================\n\n\n')
            
    #         # 停顿时间
    #         stop_hour = 0.05
    #         logger.info(f'休息{stop_hour}小时')
    #         for temp in tqdm([1 for i in range(60)]):
    #             time.sleep(stop_hour * 60)
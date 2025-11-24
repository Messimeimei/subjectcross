# from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
import fire
import torch
import json
from tqdm import tqdm
import time
from transformers import LlamaTokenizer, AutoTokenizer, set_seed
from utils import load_model
import pandas as pd

MODEL_DICT = {
    "llama2_7b_chat": "/data_share/model_hub/llama/Llama-2-7b-chat-hf",
    "llama2_13b_chat": "/data_share/model_hub/llama/Llama-2-13b-chat-hf",
    "llama2_70b_chat": "/data_share/model_hub/llama/Llama-2-70b-chat-hf",
    "llama3_8b_chat": "/data_share/model_hub/Meta-Llama-3-8B-Instruct",
    "llama3_70b_chat": "/data_share/model_hub/Meta-Llama-3-70B-Instruct",
    "mistral-7b-v0.1": "/data_share/model_hub/Mixtral/Mistral-7B-Instruct-v0.1",
    "mixtral_8x7b-v0.1": "/data_share/model_hub/Mixtral/Mixtral-8x7B-Instruct-v0.1"
}

format_v2 = {"论文与项目是否相关":"是/否","论文直接或间接研究的项目研究问题":[],"论文直接或间接采用的项目研究方法":[],"论文实现或部分实现的项目研究目标":[]}
prompt_v3 = """# 任务描述\n请根据输入的项目信息，判断论文与项目是否相关；如果相关，请依次判断项目信息中的各个研究问题是否被论文研究（包括直接和间接研究），项目信息中的各个研究方法是否被论文采用（包括直接和间接采用），项目信息中的各个研究目标论文是否实现（含部分实现）。返回内容必须为项目信息中提到的原文描述，不要删改，不要解释说明。\n请仅输出json格式，格式要求：{}\n\n# 任务输入\n【项目信息】\n标题：{}\n信息：{}\n【论文信息】\n标题：{}\n摘要：{}\n\n# 任务输出\n"""


TASK_DESCRIPTIONS = "Based on the given academic text, including the title and abstract,you first need to accurately summarize the Research Question, which must be a one-sentence WH-question.\nThen, you need to extract keywords for the following elements of your summarized research question, separately, as an expert would:\n(1)Research Object. The fundamental and core object(s) of the research.\n(2)Research Focus. The specific unknown aspect(s) of the research object that needs to be addressed.\n(3)Research Presupposition. The limiting condition(s), setting(s) or context(s) of the research question.\n(4)Question Type. The types of research questions include only 'why', 'what', and 'how'."
OUTPUT_FORMAT = "The output text should be parsed to the JSON format.\n{{\n  \"research question\": Str (WH-question): <Research Question>,\n  \"research object\": List: <research object>,\n  \"research focus\": List: <Research Focus>,\n  \"research presupposition\": List: <Research Presupposition>,\n  \"question type\": Str: <Question Type>\n}}\n\nNote that the keywords for each part should be returned in the form of a list, with each keyphrase being a concise noun or noun phrase, preferably no more than five words. "
SYS_MESSAGE = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."

def mk_message(model_name, doc):

    message = []
    if "llama2" in model_name:
        message.append({"role": "system", "content": f"{SYS_MESSAGE}"})
    # 添加任务描述与样例

    message.append({"role": "user", "content": f"{TASK_DESCRIPTIONS}\nTitle: {doc[0]}\nAbstract: {doc[1]}\n{OUTPUT_FORMAT}"})

    return message

def main(
    model_name: str="llama3_70b_chat",
    data_name: str="WoS",
    quantization: bool=False,
    max_new_tokens: int=512, #The maximum numbers of tokens to generate
    seed: int=42, #seed value for reproducibility
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int=None, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation. 
    **kwargs
):
    # 路径设置
    output_path = f"output_test_added/inference_{data_name}_{model_name}.json"
    input_path = ""

    # 读取数据集
    data = json.load(open(input_path, "r", encoding="utf-8"))

    test_set = list(data.items())
    set_seed(seed)
    model_path = MODEL_DICT[model_name]
    model = load_model(model_path, quantization)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.eval()
    solved_data = {}
    # 打开输出文件
    f_output = open(output_path, "w", encoding="utf-8")
    for doc in tqdm(test_set):
        solved_data[doc[0]] = {}
        # 生成prompt并tokenize
        message = mk_message(model_name, doc)

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
                **kwargs 
            )
        a = int(batch['attention_mask'].shape[1])

        for res_el in outputs.tolist():
            answer = tokenizer.decode(res_el[a:], skip_special_tokens=True).strip()
            solved_data[doc[0]][model_name] = answer
            
    f_output.write(json.dumps(solved_data, ensure_ascii=False))
    # 关闭输出文件
    f_output.close()


# 对csv进行推理操作
def inference_csv(
    model_name: str="llama3_70b_chat",
    quantization: bool=False,
    max_new_tokens: int=512, #The maximum numbers of tokens to generate
    seed: int=42, #seed value for reproducibility
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int=None, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation. 
    **kwargs
):
    # 数据路径
    path_dir = "/home/llmtrainer/yf_project/research_question_tree_jy/data/csv/"
    csv_path = "/home/llmtrainer/yf_project/research_question_tree_jy/data/csv/sample_data_随机抽取6项成果.csv"
    output_path = f"{path_dir}/sample_data_{model_name}.csv"
    df = pd.read_csv(csv_path, index_col=0, dtype=str)

    # 模型设置
    set_seed(seed)
    model_path = MODEL_DICT[model_name]
    model = load_model(model_path, quantization)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()
    
    new_data = []
    for index, record in df.iterrows():
        print(index)
        if index>10:
            break

        # 设置message
        message = []
        if "llama2" in model_name:
            message.append({"role": "system", "content": f"{SYS_MESSAGE}"})
        input_content = prompt_v3.format(format_v2, record["grantTitle"], record["grantInfo"], record["paperTitle"], record["paperAbstract"])
        message.append({"role": "user", "content": input_content})

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
                **kwargs 
            )
        a = int(batch['attention_mask'].shape[1])

        for res_el in outputs.tolist():
            answer = tokenizer.decode(res_el[a:], skip_special_tokens=True).strip()
            # print(answer)

        # 数据存储
        record[model_name] = answer
        new_data.append(record)
        pd.DataFrame(new_data).to_csv(f"{output_path[:-4]}_temp.csv", encoding="utf-8-sig")
    pd.DataFrame(new_data).to_csv(output_path, encoding="utf-8-sig")


if __name__ == "__main__":
    # fire.Fire(main)
    inference_csv()
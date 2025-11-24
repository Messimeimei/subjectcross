import json
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaConfig, BitsAndBytesConfig, AutoModelForCausalLM
import torch

# Function to load the main model for text generation
# def load_model(model_name, quantization):
#     if quantization:
#         print("量化")
#         bnb_config = BitsAndBytesConfig(load_in_8bit=True)
#         model = LlamaForCausalLM.from_pretrained(
#                 model_name,
#                 return_dict=True,
#                 torch_dtype=torch.float16,
#                 quantization_config=bnb_config,
#                 max_memory={0: "80GiB"},
#                 device_map='auto'
#         )
#     else:
#         print("未量化") 
#         model = LlamaForCausalLM.from_pretrained(
#             model_name,
#             return_dict=True,
#             torch_dtype=torch.float16,
#             # max_memory={0: "80GiB", 1: "80GiB"},
#             low_cpu_mem_usage=True,
#             device_map="auto"
#         )
#     return model


# Function to load the main model for text generation
def load_model(model_name_or_path, quantization, load_in_4bit=False):
    # 加载base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        return_dict=True,
        # load_in_4bit=load_in_4bit,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map='balanced' # balanced可以解决
    )
    return model

from http import HTTPStatus
import os
import dashscope
import time
import random
from langchain.output_parsers.json import parse_json_markdown
from loguru import logger
import sys


# API_KEY = "xxx" # 此处填写API Key

# 用于输出错误日志
class Log():
    def __init__(self):
        logger.remove()
        logger.add(
            sys.stdout,  # 输出到终端（类似 nohup 的屏幕输出）
            colorize=True,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{message}</cyan>"
        )

    @classmethod
    def write_log(cls, text, level="info"):
        if level == "info":
            logger.info(text)
        elif level == "error":
            logger.error(text)
        elif level == "warning":
            logger.warning(text)
        elif level == "debug":
            logger.debug(text)


def call_with_messages_deepseek(prompt_text):
    messages = [
        {'role': 'user', 'content': prompt_text}
    ]
    response = dashscope.Generation.call(
        api_key=API_KEY,
        model="deepseek-v3",  # 此处以 deepseek-r1/deepseek-v3 为例，可按需更换模型名称。
        messages=messages,
        # result_format参数不可以设置为"text"。
        result_format='message'
    )
    return response.output.choices[0].message.content


def call_with_messages_qwen_plus(prompt_text):
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': prompt_text}
    ]
    response = dashscope.Generation.call(
        api_key=API_KEY,
        model="qwen-plus",
        # 此处以qwen-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        messages=messages,
        result_format='message'
    )
    # print(response.output)
    return response.output.choices[0].message.content


def call_with_messages_qwen_plus_retry(prompt_text, default=False, default_format="", max_retries=3, wait_seconds=60):
    """
    带重试机制的调用封装
    prompt_text: 传入的提示文本
    max_retries: 最大重试次数
    wait_seconds: 出错后的等待时间（秒）
    """
    last_exception = None
    for attempt in range(1, max_retries + 1):
        try:
            return call_with_messages_qwen_plus(prompt_text)
        except Exception as e:
            last_exception = e
            Log.write_log("\n")
            Log.write_log(f"[尝试 {attempt}/{max_retries}] 调用失败: {e}")
            Log.write_log(f"PROMPR: {[prompt_text]}")
            if attempt < max_retries:
                Log.write_log(f"等待 {wait_seconds} 秒后重试...")
                time.sleep(wait_seconds + random.random())
    # 如果设计了default输出格式，避免抛出失败
    if default:
        Log.write_log(f"[尝试 {attempt}/{max_retries}] 调用失败: {e}，返回默认格式: {default_format}")
        return default_format
    # 如果多次失败，抛出最后一次异常
    raise last_exception


def call_with_messages_qwen_plus_parse(prompt_text, default=False, default_format={}, max_retries=3, wait_seconds=60):
    """
    带重试机制的调用封装，并在成功后将输出解析为 JSON 格式
    prompt_text: 传入的提示文本
    max_retries: 最大重试次数
    wait_seconds: 出错后的等待时间（秒）
    """
    last_exception = None
    for attempt in range(1, max_retries + 1):
        try:
            raw_output = call_with_messages_qwen_plus(prompt_text)
            # 解析为 JSON
            dic = parse_json_markdown(raw_output)
            return dic
        except Exception as e:
            last_exception = e
            Log.write_log(f"[尝试 {attempt}/{max_retries}] 调用失败: {e}")
            if attempt < max_retries:
                Log.write_log(f"等待 {wait_seconds} 秒后重试...")
                time.sleep(wait_seconds + random.random())
    # 如果设计了default输出格式，避免抛出失败
    if default:
        Log.write_log(f"[尝试 {attempt}/{max_retries}] 调用失败，返回默认格式: {default_format}")
        return default_format
    # 多次失败则抛出
    raise last_exception
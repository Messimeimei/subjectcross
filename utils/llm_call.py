# -*- coding: utf-8 -*-
"""
Qwen 学科相关性打分封装（支持 discipline_map 模式）
-------------------------------------------------
功能：
- 从 prompt 模板文件读取模板（title_abs / author_aff / discipline_map）
- 调用 Qwen (DashScope 兼容 API)
- 自动解析：
    - title_abs / author_aff → Python list: [("学科名/代码 名称", 分值), ...]
    - discipline_map         → JSON（对象或数组）：
         * 对象：{"某学科": [["0812 计算机科学与技术", 0.923], ...], ...}
         * 数组：[["0812 计算机科学与技术", 0.923], ...]
- ✅ discipline_map 模式用于 OpenAlex → 中国117学科映射
-------------------------------------------------
"""

import os
import re
import ast
import json
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

# ========== 环境与路径 ==========
load_dotenv()
BASE_DIR = Path(__file__).resolve().parents[1]
PROMPT_DIR = BASE_DIR / "data" / "prompts"

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen3-max")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1200"))

client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# ========== Prompt 模板加载 ==========
def _read_prompt_template(mode: str = "author_aff") -> str:
    """
    根据 mode 读取不同模板：
      - "title_abs"      →  title_abs_judge.txt
      - "author_aff"     →  author_judge.txt
      - "discipline_map" →  discipline_map.txt
    """
    if mode == "title_abs":
        filename = "title_abs_judge.txt"
    elif mode == "author_aff":
        filename = "author_judge.txt"
    elif mode == "discipline_map":
        filename = "discipline_map.txt"
    else:
        raise ValueError(f"未知 mode: {mode}")

    file_path = PROMPT_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Prompt 文件不存在: {file_path}")
    return file_path.read_text(encoding="utf-8")

# ========== 解析器（title_abs / author_aff）==========
def _parse_python_list(text: str) -> Optional[List[Tuple[str, float]]]:
    """
    解析返回的 Python list 字符串：
    目标格式：[(str, float), ...] 或 [[str, float], ...]
    """
    s = (text or "").strip()
    # 直接整体解析
    for candidate in (s, _extract_first_bracket_block(s, "[", "]")):
        if not candidate:
            continue
        try:
            res = ast.literal_eval(candidate)
            if isinstance(res, list):
                # 统一成 [ [str, float], ... ]
                norm = []
                for item in res:
                    if isinstance(item, tuple):
                        item = list(item)
                    if isinstance(item, list) and len(item) == 2:
                        name, score = item[0], float(item[1])
                        norm.append([str(name), float(score)])
                return norm
        except Exception:
            pass
    return None

def _extract_first_bracket_block(s: str, lbr: str, rbr: str) -> Optional[str]:
    """
    提取字符串中第一个由 lbr/rbr 包围的完整块（简易括号配对）。
    用于从“有解释文字包裹的返回”里抽出第一段 [ ... ] 或 { ... }。
    """
    start = s.find(lbr)
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        if s[i] == lbr:
            depth += 1
        elif s[i] == rbr:
            depth -= 1
            if depth == 0:
                return s[start:i+1]
    return None

# ========== 解析器（discipline_map）==========
def _parse_json_mapping(text: str, key: Optional[str] = None) -> Union[Dict[str, Any], List, None]:
    """
    针对 discipline_map 的解析：
    - 支持整张 JSON 对象：{"Computer Science": [[...], ...], ...}
    - 也支持仅数组：[[...], ...]
    - key 参数：若提供 key 且返回对象，优先提取对象[key]
    返回：
      - 若解析为对象且有 key：返回对象[key]（list）
      - 若解析为对象且无 key：返回该对象（dict）
      - 若解析为数组：返回该数组（list）
      - 否则返回 None
    """
    s = (text or "").strip()

    # 优先找对象 { ... }
    obj_block = _extract_first_bracket_block(s, "{", "}")
    if obj_block:
        try:
            obj = json.loads(obj_block)
            if isinstance(obj, dict):
                if key is not None and key in obj:
                    return obj[key]
                return obj
        except Exception:
            pass

    # 再尝试数组 [ ... ]
    arr_block = _extract_first_bracket_block(s, "[", "]")
    if arr_block:
        try:
            arr = json.loads(arr_block)
            if isinstance(arr, list):
                return arr
        except Exception:
            pass

    # 兜底：直接整体解析一次
    try:
        maybe = json.loads(s)
        if isinstance(maybe, (dict, list)):
            if isinstance(maybe, dict) and key is not None and key in maybe:
                return maybe[key]
            return maybe
    except Exception:
        pass

    return None

def build_prompt(
    topn_field: int,
    topn_subfield: int,
    fields_en_list: list,
    subfields_en_list: list,
    text_block: str,
    disciplines_json: list,
    disciplines_intro_json: dict,
    topn: int = 5,
    mode: str = "author_aff"
) -> str:
    """填充模板"""
    template = _read_prompt_template(mode)
    return (
        template
        .replace("{{TOPN_FIELD}}", str(topn_field))
        .replace("{{TOPN_SUBFIELD}}", str(topn_subfield))
        .replace("{{FIELDS_EN_LIST}}", json.dumps(fields_en_list or [], ensure_ascii=False, indent=2))
        .replace("{{SUBFIELDS_EN_LIST}}", json.dumps(subfields_en_list or [], ensure_ascii=False, indent=2))

        .replace("{{TEXT_BLOCK}}", text_block)
        .replace("{{DISCIPLINES_JSON}}", json.dumps(disciplines_json, ensure_ascii=False, indent=2))
        .replace("{{DISCIPLINES_INTRO_JSON}}", json.dumps(disciplines_intro_json, ensure_ascii=False, indent=2))
        .replace("{{TOPN}}", str(topn))
    )

# ========== 主函数 ==========
def call_qwen_rank(
    text_block: str,
    disciplines_json: list,
    disciplines_intro_json: dict,
    fields_en_list: list = None,
    subfields_en_list: list = None,
    topn_field: int = 5,
    topn_subfield: int = 2,
    topn: int = 5,
    mode: str = "author_aff",
    max_retries: int = 3
) -> Union[List[List[Union[str, float]]], Dict[str, Any], None]:
    """
    调用 Qwen 模型：
      - title_abs / author_aff → 返回 list: [[学科名/代码 名称, 分数], ...]
      - discipline_map         → 返回 dict 或 list：
            * dict: {"学科名": [[代码 名称, 分数], ...], ...}
            * list: [[代码 名称, 分数], ...]  （若模板设计为“单学科输入”）
    """
    # 仅选用候选学科的简介（防止 prompt 过长）
    subset_intro = {}
    for d in disciplines_json:
        code = str(d).split()[0]
        intro = disciplines_intro_json.get(code, "")
        subset_intro[code] = intro[:1000] if len(intro) > 1000 else intro

    prompt = build_prompt(topn_field, topn_subfield, fields_en_list, subfields_en_list, text_block ,disciplines_json, subset_intro, topn, mode)

    # print(f"提示词填充完毕：{prompt}")

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=QWEN_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_tokens=MAX_TOKENS,
            )
            text = resp.choices[0].message.content or ""
            # print("🧾 模型原文预览：", text[:300].replace("\n", " "))

            if mode in ("title_abs", "author_aff"):
                parsed = _parse_python_list(text)
                return parsed or []

            elif mode == "discipline_map":
                # 大模型直接返回的json，不需要解析
                # parsed = _parse_json_mapping(text, key=text_block if text_block else None)
                if isinstance(text, dict):
                    return text
                else:
                    return {}
            else:
                raise ValueError(f"未知 mode: {mode}")

        except Exception as e:
            print(f"⚠️ Qwen 调用失败（第{attempt}次）：{e}")

    return None


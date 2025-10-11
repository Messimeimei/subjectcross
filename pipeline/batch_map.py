# -*- coding: utf-8 -*-
"""
异步并发映射（async 版）：
- 读取 data/03subject_data/{csv_name}（或批量扫描整个目录）
- 异步调用 Qwen（DashScope 兼容 API）：
  7要素抽取 + 要素学科关联 + 主/交叉学科 + 交叉等级
- 将 JSON 写入 elements_crossmap_json
- 保存到 data/04map_data/{csv_name}

优化：
- asyncio + AsyncOpenAI（DashScope 兼容）
- 并发上限（WORKERS）
- QPS + RPM 双窗限速（AsyncRateLimiter）
- 入参裁剪（MAX_TITLE_CHARS / MAX_ABS_CHARS）
- max_tokens 限制
- 本地缓存（CACHE_DIR）
- 断点续跑 & 批量落盘（BATCH_FLUSH）
- 可单文件或整目录运行（.env: TARGET_CSV）
"""

import os
import re
import json
import time
import hashlib
import asyncio
from collections import deque
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from openai import AsyncOpenAI

# ===== 路径与环境 =====
BASE_DIR = Path(__file__).resolve().parents[1]
os.chdir(BASE_DIR)
print(f"📂 当前工作目录：{os.getcwd()}")

load_dotenv()

SUBJECT_DIR = BASE_DIR / "data" / "03subject_data"
OUTPUT_DIR  = BASE_DIR / "data" / "04map_data"
PROMPT_PATH = BASE_DIR / "data" / "prompts" / "research_elements_and_crossdiscipline_v2.txt"
CACHE_DIR   = BASE_DIR / "data" / "cache" / "map_api"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ===== 参数（可在 .env 调整）=====
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
QWEN_MODEL        = os.getenv("QWEN_MODEL", "qwen3-max")

# 限速与并发
QWEN_QPS          = float(os.getenv("QWEN_QPS", "2"))   # 每秒请求上限
QWEN_RPM          = int(os.getenv("QWEN_RPM", "60"))    # 每分钟请求上限
WORKERS           = int(os.getenv("WORKERS", "10"))     # 最大并发任务数（建议 <= QWEN_QPS*2）

# 落盘与输入裁剪
BATCH_FLUSH       = int(os.getenv("BATCH_FLUSH", "50"))     # 每处理多少行写盘一次
MAX_TITLE_CHARS   = int(os.getenv("MAX_TITLE_CHARS", "200"))
MAX_ABS_CHARS     = int(os.getenv("MAX_ABS_CHARS", "2500"))
MAX_TOKENS        = int(os.getenv("MAX_TOKENS", "1200"))
TOPN              = int(os.getenv("TOPN", "3"))             # 参考学科数量（消费上游列）
TARGET_CSV        = os.getenv("TARGET_CSV", "")             # 单文件运行（可选）

# ===== Async OpenAI（DashScope 兼容）=====
aclient = AsyncOpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ===== Prompt =====
def _read_prompt_template() -> str:
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"Prompt 文件不存在：{PROMPT_PATH}")
    return PROMPT_PATH.read_text(encoding="utf-8")

PROMPT_TEMPLATE = _read_prompt_template()

# ===== 工具函数 =====
def _json_from_text(text: str) -> Optional[Dict[str, Any]]:
    s = (text or "").strip()
    # 直接 JSON
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # 提取第一个 {...}
    m = re.search(r"\{.*\}", s, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None

def _ensure_list_of_disciplines(disc_list) -> List[str]:
    """从 topn_disciplines -> ['代码 名称', ...]；支持字符串或列表。"""
    out: List[str] = []
    if isinstance(disc_list, str):
        try:
            disc_list = json.loads(disc_list)
        except Exception:
            disc_list = []
    if isinstance(disc_list, list):
        for item in disc_list:
            if isinstance(item, dict) and "discipline" in item:
                val = str(item["discipline"]).strip()
                if val:
                    out.append(val)
            elif isinstance(item, str):
                val = item.strip()
                if val:
                    out.append(val)
    # 去重并截断
    seen, res = set(), []
    for d in out:
        if d not in seen:
            seen.add(d)
            res.append(d)
        if len(res) >= TOPN:
            break
    return res

def _trim(s: str, limit: int) -> str:
    s = s or ""
    return s if len(s) <= limit else s[:limit] + "…"

def build_prompt(title: str, abstract: str, disciplines: List[str]) -> str:
    return (
        PROMPT_TEMPLATE
        .replace("{{DISCIPLINES_JSON}}", json.dumps(disciplines, ensure_ascii=False, indent=2))
        .replace("{{TITLE}}", _trim(title, MAX_TITLE_CHARS))
        .replace("{{ABSTRACT}}", _trim(abstract, MAX_ABS_CHARS))
    )

# ===== 缓存（同步文件 I/O，已足够快）=====
def _hash_key(title: str, abstract: str, disciplines: List[str]) -> str:
    h = hashlib.sha1()
    h.update((title or "").encode("utf-8"))
    h.update(b"\n")
    h.update((abstract or "").encode("utf-8"))
    h.update(b"\n")
    h.update("\t".join(disciplines).encode("utf-8"))
    return h.hexdigest()

def cache_get(title: str, abstract: str, disciplines: List[str]) -> Optional[Dict[str, Any]]:
    f = CACHE_DIR / f"{_hash_key(title, abstract, disciplines)}.json"
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

def cache_put(title: str, abstract: str, disciplines: List[str], obj: Dict[str, Any]) -> None:
    f = CACHE_DIR / f"{_hash_key(title, abstract, disciplines)}.json"
    try:
        f.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

def row_done(val: Any) -> bool:
    return isinstance(val, str) and val.strip().startswith("{") and val.strip().endswith("}")

# ===== 异步限速（QPS + RPM 双窗）=====
class AsyncRateLimiter:
    def __init__(self, qps: float, rpm: int):
        self.qps = max(qps, 0.1)
        self.rpm = max(rpm, 1)
        self.lock = asyncio.Lock()
        self.sec_window = deque()
        self.min_window = deque()

    async def acquire(self):
        while True:
            async with self.lock:
                now = time.time()
                # 清理过期
                while self.sec_window and now - self.sec_window[0] >= 1.0:
                    self.sec_window.popleft()
                while self.min_window and now - self.min_window[0] >= 60.0:
                    self.min_window.popleft()

                if len(self.sec_window) < self.qps and len(self.min_window) < self.rpm:
                    self.sec_window.append(now)
                    self.min_window.append(now)
                    return

                wait_sec = max(
                    (1.0 - (now - self.sec_window[0])) if self.sec_window else 0.0,
                    (60.0 - (now - self.min_window[0])) if self.min_window else 0.0,
                )
            await asyncio.sleep(min(max(wait_sec, 0.01), 1.0))

rate_limiter = AsyncRateLimiter(QWEN_QPS, QWEN_RPM)

# ===== 单次异步调用 =====
async def call_qwen_async(prompt: str, max_retries: int = 3) -> Optional[Dict[str, Any]]:
    for attempt in range(1, max_retries + 1):
        try:
            await rate_limiter.acquire()
            resp = await aclient.chat.completions.create(
                model=QWEN_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=MAX_TOKENS,
                response_format={"type": "json_object"},  # 让模型直接返回 JSON,减少重试次数
            )
            text = (resp.choices[0].message.content or "")
            obj = _json_from_text(text)
            if obj is None:
                raise ValueError("模型输出非合法 JSON")
            return obj
        except Exception as e:
            wait = min(2 ** (attempt - 1), 8)
            print(f"⚠️ 调用失败（第{attempt}次）：{e}；{wait}s 后重试")
            await asyncio.sleep(wait)
    return None

# ===== 核心 Runner（异步版）=====
class CrossMapAsyncRunner:
    def __init__(self, batch_flush: int = BATCH_FLUSH, workers: int = WORKERS):
        self.batch_flush = batch_flush
        self.sema = asyncio.Semaphore(workers)  # 并发上限

    async def _process_one_row(self, idx: int, row: pd.Series) -> Tuple[int, str]:
        """返回 (idx, json_str)"""
        if row_done(row.get("elements_crossmap_json", "")):
            return idx, row.get("elements_crossmap_json", "")

        title = str(row.get("论文标题", "") or "")
        abstract = str(row.get("CR_摘要", "") or "")
        disciplines = _ensure_list_of_disciplines(row.get("topn_disciplines", ""))

        if not disciplines:
            return idx, json.dumps({"error": "no_disciplines"}, ensure_ascii=False)

        # 缓存命中
        cached = cache_get(title, abstract, disciplines)
        if cached is not None:
            return idx, json.dumps(cached, ensure_ascii=False)

        prompt = build_prompt(title, abstract, disciplines)

        async with self.sema:
            result = await call_qwen_async(prompt, max_retries=3)

        if result is None:
            return idx, json.dumps({"error": "model_failure"}, ensure_ascii=False)

        # 写缓存
        cache_put(title, abstract, disciplines, result)
        return idx, json.dumps(result, ensure_ascii=False)

    async def process_one_csv(self, csv_file: str) -> Tuple[int, int]:
        """处理一个 CSV 文件；返回 (成功条数, 失败条数)"""
        src = SUBJECT_DIR / csv_file if not Path(csv_file).is_absolute() else Path(csv_file)
        out_path = OUTPUT_DIR / src.name
        print(f"\n📘 处理文件：{src.name}")

        df = pd.read_csv(src)

        # 断点续跑（载入历史 results 列）
        if out_path.exists():
            try:
                old = pd.read_csv(out_path)
                if len(old) == len(df) and "elements_crossmap_json" in old.columns:
                    df["elements_crossmap_json"] = old["elements_crossmap_json"]
                    print("⏩ 续跑未完成的行")
            except Exception:
                pass

        if "elements_crossmap_json" not in df.columns:
            df["elements_crossmap_json"] = ""

        total = len(df)
        pbar = tqdm(total=total, desc=f"CrossMap {src.name}")

        # 启动任务（注意不要一次性创建几十万任务；这里按行创建即可，由信号量限制并发）
        tasks = [self._process_one_row(i, row) for i, row in df.iterrows()]

        done_cnt, fail_cnt = 0, 0
        flush_counter = 0

        for coro in asyncio.as_completed(tasks):
            try:
                idx, json_str = await coro
                df.at[idx, "elements_crossmap_json"] = json_str
                # 粗略判断成功/失败
                try:
                    obj = json.loads(json_str)
                    if isinstance(obj, dict) and "error" in obj:
                        fail_cnt += 1
                    else:
                        done_cnt += 1
                except Exception:
                    fail_cnt += 1
            except Exception as e:
                fail_cnt += 1
                # 如果连 idx 都拿不到，就没法回填；这里仅记录
                print("❌ 异常：", e)

            pbar.update(1)
            flush_counter += 1

            # 周期性落盘
            if flush_counter % self.batch_flush == 0:
                df.to_csv(out_path, index=False, encoding="utf-8-sig")

        # 最终落盘
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"✅ 完成 {src.name}：成功 {done_cnt}，失败 {fail_cnt}")
        return done_cnt, fail_cnt

    async def process_all(self):
        """批量处理整个目录"""
        csv_files = sorted([f for f in os.listdir(SUBJECT_DIR) if f.endswith(".csv")])
        summary = []
        for name in csv_files:
            d, f = await self.process_one_csv(name)
            summary.append((name, d, f))

        print("\n📊 全部处理完成：")
        for name, d, f in summary:
            print(f" - {name}: 成功 {d}, 失败 {f}")

# ===== CLI =====
async def _amain():
    runner = CrossMapAsyncRunner(batch_flush=BATCH_FLUSH, workers=WORKERS)
    if TARGET_CSV:
        await runner.process_one_csv(TARGET_CSV)
    else:
        await runner.process_all()

if __name__ == "__main__":
    asyncio.run(_amain())

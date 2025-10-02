# direction2discipline.py
# 功能：
# 1) 读取 ../data/zh_disciplines_with_code.csv（117 个一级学科清单）。
# 2) 输入方向字符串 (如 "1001 Basic Medicine; 1002 Clinical Medicine; 0710 Biology")，
#    提取前面的学科代码，匹配到中文清单。
# 3) 返回 117 学科分数字典 {代码+中文: 分数}，命中的均分，其余为 0。

import json
from typing import Dict

class Direction2Discipline:
    def __init__(self, disciplines_file: str):
        self.code2name = self._load_disciplines(disciplines_file)

    def _load_disciplines(self, path: str) -> Dict[str, str]:
        """读取117学科清单，支持空格分隔"""
        code2name = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    code, name = parts[0], parts[1]
                    code2name[code] = name
        return code2name

    def direction_to_scores(self, direction_str: str) -> Dict[str, float]:
        """
        输入: "1001 Basic Medicine; 1002 Clinical Medicine; 0710 Biology"
        输出: {代码+中文: 分数}，覆盖全部117学科
        """
        scores = {f"{c} {n}": 0.0 for c, n in self.code2name.items()}
        if not direction_str:
            return scores

        # 拆分输入
        parts = [p.strip() for p in direction_str.split(";") if p.strip()]
        matched_codes = []

        for p in parts:
            code = p.split()[0]  # 取第一个空格前的部分，例如 "1001"
            if code in self.code2name:
                matched_codes.append(code)

        # 均分
        if matched_codes:
            weight = 1.0 / len(matched_codes)
            for c in matched_codes:
                scores[f"{c} {self.code2name[c]}"] = weight

        return scores


# ========= 使用示例 =========
if __name__ == "__main__":
    disc_file = "../data/zh_disciplines_with_code.csv"
    d2d = Direction2Discipline(disc_file)

    test_str = "1001 Basic Medicine; 1002 Clinical Medicine; 0710 Biology"
    result = d2d.direction_to_scores(test_str)

    print(json.dumps(result, ensure_ascii=False, indent=2))

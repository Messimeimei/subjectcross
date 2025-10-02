# Created by Messimeimei
# Created at 2025/9/29

"""用于读取学科介绍文件，为中文学科介绍文件添加代码方便与Incites数据库对应"""

import json
import pandas as pd

def get_zn_discipline():
    # 读取中文学科介绍文件，获取中文学科

    with open("../zh_discipline_intro.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # 获取所有的键（一级学科名称）
    keys = list(data.keys())

    # 转成 DataFrame
    df = pd.DataFrame(keys, columns=["一级学科"])

    # 保存到 CSV
    df.to_csv("../zh_disciplines.csv", index=False, encoding="utf-8-sig")

    print("已保存 zh_disciplines.csv，包含 %d 个一级学科名称。" % len(keys))

def create_disciplineinto_with_code():
    # 为原始的中文学科映射文件添加代码，方便日后使用，与英文学科对应上

    import json
    import pandas as pd

    # 读取 csv
    df = pd.read_csv("../zh_disciplines_with_code.csv", sep=" ", header=None, names=["code", "name"])

    # 保证 code 是字符串 & 保留前导零
    df["code"] = df["code"].astype(str).str.zfill(4)

    # 读取原始 JSON
    with open("../zh_discipline_intro.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # 合并
    new_data = {}
    for _, row in df.iterrows():
        code, name = row["code"], row["name"]
        intro = data.get(name, "")
        new_data[code] = {
            "name": name,
            "intro": intro
        }

    with open("../zh_discipline_intro_with_code.json", "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print("✅ 已生成 zh_discipline_intro_with_code.json，学科代码保留4位")


if __name__ == '__main__':
    create_disciplineinto_with_code()
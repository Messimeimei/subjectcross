# @Time    : 2024/07/05
# @Author  : Yifan 
# @Purpose : 评价模型训练效果

import json
from rouge_score import rouge_scorer
from rouge_chinese import Rouge
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from bert_score import score
import jieba
import numpy as np
import re
from loguru import logger
import pickle

# 计算平均值
def list_average(data_list):
    if len(data_list) == 0:
        return 0
    else:
        return sum(data_list) / len(data_list)

# 读jsonlines文件
def read_jsonl_by_line(path):
    f = open(path, "r")
    record = f.readline()
    while record:
        record = record.strip()
        dic = json.loads(record)
        yield dic
        record = f.readline()
    f.close()

# rouge score 原版
def rouge_score_origin(reference, pred, use_stemmer=True):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, pred)
    print(scores)
    return scores

# rouge score 中文版
def rouge_score_chinese(reference, pred):
    # 为空的情况
    if reference == "" or pred == "":
        if reference == pred:
            return {'rouge-1': {'r': 1.0, 'p': 1.0, 'f': 1.0}, 'rouge-2': {'r': 1.0, 'p': 1.0, 'f': 1.0}, 'rouge-l': {'r': 1.0, 'p': 1.0, 'f': 1.0}}
        else:
            return {'rouge-1': {'r': 0.0, 'p': 0.0, 'f': 0.0}, 'rouge-2': {'r': 0.0, 'p': 0.0, 'f': 0.0}, 'rouge-l': {'r': 0.0, 'p': 0.0, 'f': 0.0}}
    # 正常情况
    pred = ' '.join(jieba.cut(pred)) 
    reference = ' '.join(jieba.cut(reference))
    rouge = Rouge()
    scores = rouge.get_scores(pred, reference)
    # print(scores)
    return scores[0]

def metric_prf_score(truth_list, pred_list, metric_function=rouge_score_chinese):
    """
    基于指定的metric：比如 rouge、jaccard、bertscore等，
    首先计算pred_str和每一个truth_str的metric score，
    然后，取最大值作为该pred_str的最终得分（取值范围[0,1]）
    最后，在此基础上计算PRF：e.g. P_rouge = (0.3+0.1+1+0.8) / (1+1+1+1)
    :param: truth_list: string 真值
    :param: pred_list: string 预测值
    :param: metric_function: 计算string得分的指标函数
    :param: use_stemmer: bool 是否进行stemming
    :return: score_results: list(float)
    """
    # 准确率
    pred_max_score_dict = {}
    for pred_str in pred_list:
        scores = []
        for truth_str in truth_list:
            score = metric_function(truth_str, pred_str)
            if metric_function == rouge_score_origin:
                scores.append(score['rougeL'].fmeasure)
            elif metric_function == rouge_score_chinese:
                scores.append(score['rouge-l']['f'])
            # if metric_function == jaccard_score:
            #     scores.append(score)
        pred_max_score_dict[pred_str] = max(scores) if len(scores) != 0 else 0
    total_value = sum(list(pred_max_score_dict.values()))
    metric_P = total_value / len(pred_list) if len(pred_list) != 0 else 0

    # 召回率
    pred_max_score_dict = {}
    for truth_str in truth_list:
        scores = []
        for pred_str in pred_list:
            score = metric_function(truth_str, pred_str)
            if metric_function == rouge_score_origin:
                scores.append(score['rougeL'].fmeasure)
            elif metric_function == rouge_score_chinese:
                scores.append(score['rouge-l']['f'])
            # if metric_function == jaccard_score:
            #     scores.append(score)
        pred_max_score_dict[truth_str] = max(scores) if len(scores) != 0 else 0
    total_value = sum(list(pred_max_score_dict.values()))
    metric_R = total_value / len(truth_list) if len(truth_list) != 0 else 0

    # 调和平均值
    metric_F = 2 * metric_P * metric_R / (metric_P + metric_R) if metric_P + metric_R != 0 else 0
    return [metric_P, metric_R, metric_F]


def bert_score_evaluate(reference, pred):
    # 指定预训练的中文BERT模型
    model_type = "bert-base-chinese"

    # 计算BERTScore
    P, R, F1 = score(pred, reference, model_type=model_type, lang="zh", verbose=True)

    # 输出结果
    # for i in range(len(references)):
    #     print(f"参考文本: {references[i]}")
    #     print(f"候选文本: {preds[i]}")
    #     print(f"精确度(Precision): {P[i].item():.4f}")
    #     print(f"召回率(Recall): {R[i].item():.4f}")
    #     print(f"F1分数: {F1[i].item():.4f}")
    #     print()
    
    # return [P, R, F1]
    return {"p": P, "r": R, "f": F1}


# 问题方法目标标注：检查各个字段是否都有，如果没有的话增补为空
def result_format_check(dic):
    for field in ["研究背景", "研究意义"]:
        if field not in dic:
            dic[field] = ""
    if "研究内容" not in dic:
        dic["研究内容"] = {}
    for field in ["研究问题", "研究方法", "研究目标"]:
        if field not in dic["研究内容"]:
            dic["研究内容"][field] = [""]
    return dic


# 问题方法目标标注：分别计算 rouge 和 bertsocre
def get_evaluate_value(path_test_jsonl, metric="rouge"):
    test_data = read_jsonl_by_line(path_test_jsonl)
    evaluate_dic = {"研究背景": [], "研究问题": [], "研究方法": [], "研究目标": [], "研究意义": []}
    for record in test_data:
        print(record["conversation_id"], "="*10)
        reference = json.loads(record["conversation"][0]["assistant"])
        pred = json.loads(record["conversation"][0]["pred"])
        pred = result_format_check(pred)
        # print(reference, "\n", pred)

        if metric=="rouge":
            # # rouge 方法 输出版
            # print("研究背景", rouge_score_chinese(reference["研究背景"], pred["研究背景"]))
            # print("研究问题", metric_prf_score(reference["研究内容"]["研究问题"], pred["研究内容"]["研究问题"]))
            # print("研究方法", metric_prf_score(reference["研究内容"]["研究方法"], pred["研究内容"]["研究方法"]))
            # print("研究目标", metric_prf_score(reference["研究内容"]["研究目标"], pred["研究内容"]["研究目标"]))
            # print("研究意义", rouge_score_chinese(reference["研究意义"], pred["研究意义"]))
            
            # rouge 方法 v2 再算平均版
            for field in ["研究背景", "研究意义"]:
                rouge_l = rouge_score_chinese(reference[field], pred[field])["rouge-l"]
                evaluate_dic[field].append([rouge_l["p"], rouge_l["r"], rouge_l["f"]])
            for field in ["研究问题", "研究方法", "研究目标"]:
                rouge_l = metric_prf_score(reference["研究内容"][field], pred["研究内容"][field])
                evaluate_dic[field].append(rouge_l)
        elif metric == "bertscore":
            # bert score 方法 输出版
            # print("研究背景", bert_score_evaluate(reference["研究背景"], pred["研究背景"]))
            # print("研究问题", bert_score_evaluate(reference["研究内容"]["研究问题"], pred["研究内容"]["研究问题"]))
            # print("研究方法", bert_score_evaluate(reference["研究内容"]["研究方法"], pred["研究内容"]["研究方法"]))
            # print("研究目标", bert_score_evaluate(reference["研究内容"]["研究目标"], pred["研究内容"]["研究目标"]))
            # print("研究意义", bert_score_evaluate(reference["研究意义"], pred["研究意义"]))
            
            # bert score 方法 再算平均版
            for field in ["研究背景", "研究意义"]:
                evaluate_dic[field].append(bert_score_evaluate([reference[field]], [pred[field]]))
            for field in ["研究问题", "研究方法", "研究目标"]:
                evaluate_dic[field].append(bert_score_evaluate([str(reference["研究内容"][field])], [str(pred["研究内容"][field])]))

    for field, value_list in evaluate_dic.items():
        print(field, np.mean([row[0] for row in value_list]), np.mean([row[1] for row in value_list]), np.mean([row[2] for row in value_list]))
    return evaluate_dic


# 论文项目匹配标注：检查各个字段是否都有，如果没有的话增补为空
def result_format_check_matchness(dic):
    default_dic = {
        "论文与项目是否相关": "",     
        "研究问题": {"核心问题": [], "相关问题": []},  
        "研究方法": [],  
        "研究目标": {"实现": [], "部分实现": []}
    }
    # 第一层确认
    for field in default_dic.keys():
        if field not in dic:
            dic[field] = default_dic[field]
    # 第二层确认
    for field in default_dic["研究问题"].keys():
        if field not in dic["研究问题"]:
            dic["研究问题"][field] = []
    for field in default_dic["研究目标"].keys():
        if field not in dic["研究目标"]:
            dic["研究目标"][field] = []
    return dic

# 论文项目匹配标注：检查各个字段是否都有，如果没有的话增补为空
def result_format_check_matchness_2types(dic):
    default_dic = {
        "论文与项目是否相关": "",     
        "论文直接或间接研究的项目研究问题": [],  
        "论文直接或间接采用的项目研究方法": [],  
        "论文实现或部分实现的项目研究目标": []
    }
    # 确认
    for field in default_dic.keys():
        if field not in dic:
            dic[field] = default_dic[field]
    return dic


# 论文项目匹配标注：计算预测准确率
# 对邻近类别有包容
def evaluate_accuracy_part(path_test_jsonl):
    test_data = read_jsonl_by_line(path_test_jsonl)
    evaluate_dic = {"总体": [], "研究问题": [], "研究方法": [], "研究目标": []}
    for record in test_data:
        print(record["conversation_id"], "="*10)
        # 加载与格式校对
        reference = json.loads(record["conversation"][0]["assistant"])
        pred = json.loads(record["conversation"][0]["pred"])
        pred = result_format_check_matchness(pred)
        p = re.search("\n信息：(.*?)\n【论文信息】\n标题：", record["conversation"][0]["human"])
        grant_info = json.loads(p.group(1).replace("\'", "\""))
        # print(reference, "\n", pred)

        # 总体
        if pred["论文与项目是否相关"] == reference["论文与项目是否相关"]:
            evaluate_dic["总体"].append(1)
        else:
            evaluate_dic["总体"].append(0)
        # 研究问题
        problem_list = grant_info["研究内容"]["研究问题"]
        problem_ref, problem_pred = [], []
        for problem in problem_list:
            if problem in reference["研究问题"]["核心问题"]:
                problem_ref.append(2)
            elif problem in reference["研究问题"]["相关问题"]:
                problem_ref.append(1)
            else:
                problem_ref.append(0)
            if problem in pred["研究问题"]["核心问题"]:
                problem_pred.append(2)
            elif problem in pred["研究问题"]["相关问题"]:
                problem_pred.append(1)
            else:
                problem_pred.append(0)
        temp_problem_match = [1 - abs(problem_ref[i] - problem_pred[i]) / 2 for i in range(len(problem_list))]
        evaluate_dic["研究问题"].append(list_average(temp_problem_match))
        # 研究方法
        method_list = grant_info["研究内容"]["研究方法"]
        temp_method_match = []
        for method in method_list:
            if method in reference["研究方法"] and method in pred["研究方法"]:
                temp_method_match.append(1)
            elif method not in reference["研究方法"] and method not in pred["研究方法"]:
                temp_method_match.append(1)
            else:
                temp_method_match.append(0)
        evaluate_dic["研究方法"].append(list_average(temp_method_match))
        # 研究目标
        object_list = grant_info["研究内容"]["研究目标"]
        object_ref, object_pred = [], []
        for object in object_list:
            if object in reference["研究目标"]["实现"]:
                object_ref.append(2)
            elif object in reference["研究目标"]["部分实现"]:
                object_ref.append(1)
            else:
                object_ref.append(0)
            if object in pred["研究目标"]["实现"]:
                object_pred.append(2)
            elif object in pred["研究目标"]["部分实现"]:
                object_pred.append(1)
            else:
                object_pred.append(0)
        temp_object_match = [1 - abs(object_ref[i] - object_pred[i]) / 2 for i in range(len(object_list))]
        evaluate_dic["研究目标"].append(list_average(temp_object_match))        
        if len(evaluate_dic["总体"]) >= 20:
            break

    for field, value_list in evaluate_dic.items():
        print(field, np.mean(value_list))
    return evaluate_dic

# 论文项目匹配标注：计算预测准确率
# 分小类计算准确率、召回率和调和平均值
def evaluate_prf_v1(path_test_jsonl):
    test_data = read_jsonl_by_line(path_test_jsonl)
    evaluate_dic = {"总体": {"predict": [], "reference": []},
                    "研究问题": {"predict": [], "reference": []},
                    "研究方法": {"predict": [], "reference": []},
                    "研究目标": {"predict": [], "reference": []},
                    }
    for record in test_data:
        print(record["conversation_id"], "="*10)
        # 加载与格式校对
        reference = json.loads(record["conversation"][0]["assistant"])
        pred = json.loads(record["conversation"][0]["pred"])
        pred = result_format_check_matchness(pred)
        p = re.search("\n信息：([\s\S]*?)\n【论文信息】\n标题：", record["conversation"][0]["human"])
        grant_info = json.loads(p.group(1).replace("\'", "\""))

        # 总体
        evaluate_dic["总体"]["predict"].append("相关" if pred["论文与项目是否相关"]=="相关" else "无关")
        evaluate_dic["总体"]["reference"].append("相关" if reference["论文与项目是否相关"]=="相关" else "无关")
        # 研究问题
        problem_list = grant_info["研究内容"]["研究问题"]
        for problem in problem_list:
            if problem in reference["研究问题"]["核心问题"]:
                evaluate_dic["研究问题"]["reference"].append("核心问题")
            elif problem in reference["研究问题"]["相关问题"]:
                evaluate_dic["研究问题"]["reference"].append("相关问题")
            else:
                evaluate_dic["研究问题"]["reference"].append("无关问题")
            if problem in pred["研究问题"]["核心问题"]:
                evaluate_dic["研究问题"]["predict"].append("核心问题")
            elif problem in pred["研究问题"]["相关问题"]:
                evaluate_dic["研究问题"]["predict"].append("相关问题")
            else:
                evaluate_dic["研究问题"]["predict"].append("无关问题")
        # 研究方法
        method_list = grant_info["研究内容"]["研究方法"]
        for method in method_list:
            evaluate_dic["研究方法"]["predict"].append("采用" if method in pred["研究方法"] else "未采用")
            evaluate_dic["研究方法"]["reference"].append("采用" if method in reference["研究方法"] else "未采用")
        # 研究目标
        object_list = grant_info["研究内容"]["研究目标"]
        for object in object_list:
            if object in reference["研究目标"]["实现"]:
                evaluate_dic["研究目标"]["reference"].append("实现")
            elif object in reference["研究目标"]["部分实现"]:
                evaluate_dic["研究目标"]["reference"].append("部分实现")
            else:
                evaluate_dic["研究目标"]["reference"].append("未实现")
            if object in pred["研究目标"]["实现"]:
                evaluate_dic["研究目标"]["predict"].append("实现")
            elif object in pred["研究目标"]["部分实现"]:
                evaluate_dic["研究目标"]["predict"].append("部分实现")
            else:
                evaluate_dic["研究目标"]["predict"].append("未实现")
        # 停止机制
        # if len(evaluate_dic["总体"]["predict"]) >= 20:
        #     break
    
    for field in ["总体", "研究问题", "研究方法", "研究目标"]:
        precision, recall, f1, support = precision_recall_fscore_support(evaluate_dic[field]["reference"], evaluate_dic[field]["predict"], average=None)
        # print(field, "precision:", precision, "recall:", recall, "f1:", f1)
        print(field,"\n", classification_report(evaluate_dic[field]["reference"], evaluate_dic[field]["predict"]))

def evaluate_rouge(path_test_jsonl):
    test_data = read_jsonl_by_line(path_test_jsonl)
    evaluate_dic = {
                    "研究问题": {"核心问题": [], "相关问题": []},
                    "研究方法": [],
                    "研究目标": {"实现": [], "部分实现": []},
                    }
    for record in test_data:
        print(record["conversation_id"], "="*10)
        # 加载与格式校对
        reference = json.loads(record["conversation"][0]["assistant"])
        pred = json.loads(record["conversation"][0]["pred"])
        pred = result_format_check_matchness(pred)
        p = re.search("\n信息：(.*?)\n【论文信息】\n标题：", record["conversation"][0]["human"])
        # grant_info = json.loads(p.group(1).replace("\'", "\""))

        # 研究问题
        for matchness in ["核心问题", "相关问题"]:
            rouge_l = metric_prf_score(reference["研究问题"][matchness], pred["研究问题"][matchness])
            evaluate_dic["研究问题"][matchness].append(rouge_l)
        # 研究方法
        rouge_l = metric_prf_score(reference["研究方法"], pred["研究方法"])
        evaluate_dic["研究方法"].append(rouge_l)
        # 研究目标
        for matchness in ["实现", "部分实现"]:
            rouge_l = metric_prf_score(reference["研究目标"][matchness], pred["研究目标"][matchness])
            evaluate_dic["研究目标"][matchness].append(rouge_l)

        # 停止机制
        if len(evaluate_dic["研究方法"]) >= 20:
            break
    
    # 研究问题
    for field, value_list in evaluate_dic["研究问题"].items():
        print("研究问题", field, np.mean([row[0] for row in value_list]), np.mean([row[1] for row in value_list]), np.mean([row[2] for row in value_list]))
    # 研究目标
    for field, value_list in evaluate_dic["研究目标"].items():
        print("研究目标", field, np.mean([row[0] for row in value_list]), np.mean([row[1] for row in value_list]), np.mean([row[2] for row in value_list]))
    # 研究方法
    value_list = evaluate_dic["研究方法"]
    print("研究方法", np.mean([row[0] for row in value_list]), np.mean([row[1] for row in value_list]), np.mean([row[2] for row in value_list]))

# 论文项目匹配标注：计算预测准确率
# 分小类计算准确率、召回率和调和平均值
def evaluate_prf_v2(path_test_jsonl):
    test_data = read_jsonl_by_line(path_test_jsonl)
    evaluate_dic = {"总体": {"predict": [], "reference": []},
                    "研究问题": {"predict": [], "reference": []},
                    "研究方法": {"predict": [], "reference": []},
                    "研究目标": {"predict": [], "reference": []},
                    }
    for record in test_data:
        print(record["conversation_id"], "="*10)
        # 加载与格式校对
        reference = json.loads(record["conversation"][0]["assistant"])
        pred = json.loads(record["conversation"][0]["pred"])
        pred = result_format_check_matchness(pred)
        p = re.search("\n信息：([\s\S]*?)\n【论文信息】\n标题：", record["conversation"][0]["human"])
        grant_info = json.loads(p.group(1).replace("\'", "\""))

        # 总体
        evaluate_dic["总体"]["predict"].append(pred["论文与项目是否相关"])
        evaluate_dic["总体"]["reference"].append(reference["论文与项目是否相关"])
        # if pred["论文与项目是否相关"]=="高度相关":
        #     evaluate_dic["总体"]["predict"].append("高度相关")
        # elif pred["论文与项目是否相关"]=="相关":
        #     evaluate_dic["总体"]["predict"].append("相关")
        # else:
        #     evaluate_dic["总体"]["predict"].append("不相关")
        # if reference["论文与项目是否相关"]=="高度相关":
        #     evaluate_dic["总体"]["reference"].append("高度相关")
        # elif reference["论文与项目是否相关"]=="相关":
        #     evaluate_dic["总体"]["reference"].append("相关")
        # else:
        #     evaluate_dic["总体"]["reference"].append("不相关")
        # 研究问题
        problem_list = grant_info["研究内容"]["研究问题"]
        for problem in problem_list:
            if problem in reference["研究问题"]["直接研究"]:
                evaluate_dic["研究问题"]["reference"].append("直接研究")
            elif problem in reference["研究问题"]["间接研究"]:
                evaluate_dic["研究问题"]["reference"].append("间接研究")
            else:
                evaluate_dic["研究问题"]["reference"].append("未研究")
            if problem in pred["研究问题"]["直接研究"]:
                evaluate_dic["研究问题"]["predict"].append("直接研究")
            elif problem in pred["研究问题"]["间接研究"]:
                evaluate_dic["研究问题"]["predict"].append("间接研究")
            else:
                evaluate_dic["研究问题"]["predict"].append("未研究")
        # 研究方法
        method_list = grant_info["研究内容"]["研究方法"]
        for method in method_list:
            evaluate_dic["研究方法"]["predict"].append("采用" if method in pred["研究方法"] else "未采用")
            evaluate_dic["研究方法"]["reference"].append("采用" if method in reference["研究方法"] else "未采用")
        # 研究目标
        object_list = grant_info["研究内容"]["研究目标"]
        for object in object_list:
            if object in reference["研究目标"]["完全实现"]:
                evaluate_dic["研究目标"]["reference"].append("完全实现")
            elif object in reference["研究目标"]["部分实现"]:
                evaluate_dic["研究目标"]["reference"].append("部分实现")
            else:
                evaluate_dic["研究目标"]["reference"].append("完全未实现")
            if object in pred["研究目标"]["完全实现"]:
                evaluate_dic["研究目标"]["predict"].append("完全实现")
            elif object in pred["研究目标"]["部分实现"]:
                evaluate_dic["研究目标"]["predict"].append("部分实现")
            else:
                evaluate_dic["研究目标"]["predict"].append("完全未实现")
        # 停止机制
        # if len(evaluate_dic["总体"]["predict"]) >= 20:
        #     break
    
    for field in ["总体", "研究问题", "研究方法", "研究目标"]:
        precision, recall, f1, support = precision_recall_fscore_support(evaluate_dic[field]["reference"], evaluate_dic[field]["predict"], average=None)
        # print(field, "precision:", precision, "recall:", recall, "f1:", f1)
        print(field,"\n", classification_report(evaluate_dic[field]["reference"], evaluate_dic[field]["predict"]))


# 论文项目匹配标注：计算预测准确率
# 分小类计算准确率、召回率和调和平均值，只分两类
def evaluate_prf_v3(path_test_jsonl):
    test_data = read_jsonl_by_line(path_test_jsonl)
    evaluate_dic = {"总体": {"predict": [], "reference": []},
                    "研究问题": {"predict": [], "reference": []},
                    "研究方法": {"predict": [], "reference": []},
                    "研究目标": {"predict": [], "reference": []},
                    }
    for record in test_data:
        # print(record["conversation_id"], "="*10)
        # 加载与格式校对
        reference = json.loads(record["conversation"][0]["assistant"])
        pred = json.loads(record["conversation"][0]["pred"])
        pred = result_format_check_matchness_2types(pred)
        # print(pred)
        p = re.search("\n信息：([\s\S]*?)\n【论文信息】\n标题：", record["conversation"][0]["human"])
        grant_info = json.loads(p.group(1).replace("\'", "\""))

        # 总体
        evaluate_dic["总体"]["predict"].append(pred["论文与项目是否相关"])
        evaluate_dic["总体"]["reference"].append(reference["论文与项目是否相关"])
        # 研究问题
        problem_list = grant_info["研究内容"]["研究问题"]
        for problem in problem_list:
            evaluate_dic["研究问题"]["reference"].append("研究" if problem in reference["论文直接或间接研究的项目研究问题"] else "未研究")
            evaluate_dic["研究问题"]["predict"].append("研究" if problem in pred["论文直接或间接研究的项目研究问题"] else "未研究")
        # 研究方法
        method_list = grant_info["研究内容"]["研究方法"]
        for method in method_list:
            evaluate_dic["研究方法"]["predict"].append("采用" if method in pred["论文直接或间接采用的项目研究方法"] else "未采用")
            evaluate_dic["研究方法"]["reference"].append("采用" if method in reference["论文直接或间接采用的项目研究方法"] else "未采用")
        # 研究目标
        object_list = grant_info["研究内容"]["研究目标"]
        for object in object_list:
            evaluate_dic["研究目标"]["predict"].append("实现" if object in pred["论文实现或部分实现的项目研究目标"] else "未实现")
            evaluate_dic["研究目标"]["reference"].append("实现" if object in reference["论文实现或部分实现的项目研究目标"] else "未实现")
        # 停止机制
        if len(evaluate_dic["总体"]["predict"]) >= 85:
            break
        # if int(record["conversation_id"]) >=1073:
        #     break

    for field in ["总体", "研究问题", "研究方法", "研究目标"]:
        precision, recall, f1, support = precision_recall_fscore_support(evaluate_dic[field]["reference"], evaluate_dic[field]["predict"], average=None)
        # print(field, "precision:", precision, "recall:", recall, "f1:", f1)
        print(field,"\n", classification_report(evaluate_dic[field]["reference"], evaluate_dic[field]["predict"]))

# 论文项目匹配标注：计算预测准确率
# 分小类计算准确率、召回率和调和平均值，只分两类，同时包括positive和negative
def evaluate_prf_v4(path_test_jsonl):
    test_data = read_jsonl_by_line(path_test_jsonl)
    evaluate_dic = {"总体": {"predict": [], "reference": []},
                    "研究问题": {"predict": [], "reference": []},
                    "研究方法": {"predict": [], "reference": []},
                    "研究目标": {"predict": [], "reference": []},
                    }
    for record in test_data:
        print(record["conversation_id"], "="*10)
        # 加载与格式校对
        reference = json.loads(record["conversation"][0]["assistant"])
        pred = json.loads(record["conversation"][0]["pred"])
        pred = result_format_check_matchness_2types(pred)
        print(pred)
        p = re.search("\n信息：([\s\S]*?)\n【论文信息】\n标题：", record["conversation"][0]["human"])
        grant_info = json.loads(p.group(1).replace("\'", "\""))

        # 总体
        evaluate_dic["总体"]["predict"].append(pred["论文与项目是否相关"])
        evaluate_dic["总体"]["reference"].append(reference["论文与项目是否相关"])
        # 研究问题
        problem_list = grant_info["研究内容"]["研究问题"]
        for problem in problem_list:
            if problem in reference["研究问题"]["直接或间接研究"]:
                evaluate_dic["研究问题"]["reference"].append("直接或间接研究")
            elif problem in reference["研究问题"]["完全未研究"]:
                evaluate_dic["研究问题"]["reference"].append("完全未研究")
            else:
                evaluate_dic["研究问题"]["reference"].append("错误结果")
            if problem in pred["研究问题"]["直接或间接研究"]:
                evaluate_dic["研究问题"]["predict"].append("直接或间接研究")
            elif problem in pred["研究问题"]["完全未研究"]:
                evaluate_dic["研究问题"]["predict"].append("完全未研究")
            else:
                evaluate_dic["研究问题"]["predict"].append("错误结果")
        # 研究方法
        method_list = grant_info["研究内容"]["研究方法"]
        for method in method_list:
            if method in reference["研究方法"]["直接或间接采用"]:
                evaluate_dic["研究方法"]["reference"].append("直接或间接采用")
            elif method in reference["研究方法"]["完全未采用"]:
                evaluate_dic["研究方法"]["reference"].append("完全未采用")
            else:
                evaluate_dic["研究方法"]["reference"].append("错误结果")
            if method in pred["研究方法"]["直接或间接采用"]:
                evaluate_dic["研究方法"]["predict"].append("直接或间接采用")
            elif method in pred["研究方法"]["完全未采用"]:
                evaluate_dic["研究方法"]["predict"].append("完全未采用")
            else:
                evaluate_dic["研究方法"]["predict"].append("错误结果")
        # 研究目标
        object_list = grant_info["研究内容"]["研究目标"]
        for object in object_list:
            if object in reference["研究目标"]["完全或部分实现"]:
                evaluate_dic["研究目标"]["reference"].append("完全或部分实现")
            elif object in reference["研究目标"]["完全未实现"]:
                evaluate_dic["研究目标"]["reference"].append("完全未实现")
            else:
                evaluate_dic["研究目标"]["reference"].append("错误结果")
            if object in pred["研究目标"]["完全或部分实现"]:
                evaluate_dic["研究目标"]["predict"].append("完全或部分实现")
            elif object in pred["研究目标"]["完全未实现"]:
                evaluate_dic["研究目标"]["predict"].append("完全未实现")
            else:
                evaluate_dic["研究目标"]["predict"].append("错误结果")

        # 停止机制
        # if len(evaluate_dic["总体"]["predict"]) >= 40:
        #     break
    
    for field in ["总体", "研究问题", "研究方法", "研究目标"]:
        precision, recall, f1, support = precision_recall_fscore_support(evaluate_dic[field]["reference"], evaluate_dic[field]["predict"], average=None)
        # print(field, "precision:", precision, "recall:", recall, "f1:", f1)
        print(field,"\n", classification_report(evaluate_dic[field]["reference"], evaluate_dic[field]["predict"]))


def rouge_and_bertscore_as_whole(path_test_jsonl):
    test_data = read_jsonl_by_line(path_test_jsonl)
    results_avg = {}
    # 3种rouge
    metrics_list = ["rouge-1", "rouge-2", "rouge-l"]
    rouge = Rouge(metrics=metrics_list)
    results = []
    for index, record in enumerate(test_data):
        if record["conversation_id"] <= 180:
            continue
        pred = ' '.join(jieba.cut(record["conversation"][0]["pred"])) 
        reference = ' '.join(jieba.cut(record["conversation"][0]["assistant"]))
        scores = rouge.get_scores(pred, reference)
        results.append(scores[0])
    for metric in metrics_list:
        results_avg[metric] = {}
        for prf in ["p", "r", "f"]:
            results_avg[metric][prf] = np.mean([record[metric][prf] for record in results])
    # bertscore
    results = []
    test_data = read_jsonl_by_line(path_test_jsonl)
    for index, record in enumerate(test_data):
        if record["conversation_id"] <= 180:
            continue
        # if index > 10:
        #     break
        pred = [record["conversation"][0]["pred"]]
        reference = [record["conversation"][0]["assistant"]]
        scores = bert_score_evaluate(reference, pred)
        results.append(scores)
    metric = "bert-score"
    results_avg[metric] = {}
    for prf in ["p", "r", "f"]:
        results_avg[metric][prf] = np.mean([record[prf] for record in results])
    # print(results_avg)
    return results_avg

def rouge_and_bertscore_each_field(path_test_jsonl):
    test_data = read_jsonl_by_line(path_test_jsonl)
    results_avg = {}
    field_list = ["研究背景", "研究问题", "研究方法", "研究目标", "研究意义"]
    # 3种rouge
    metrics_list = ["rouge-1", "rouge-2", "rouge-l"]
    rouge = Rouge(metrics=metrics_list)
    results = {field: [] for field in field_list}
    for index, record in enumerate(test_data):
        if record["conversation_id"] <= 180:
            continue
        # 数据读入
        reference = json.loads(record["conversation"][0]["assistant"])
        pred = json.loads(record["conversation"][0]["pred"])
        pred = result_format_check(pred)
        # 分字段评价
        for field in field_list:
            if field in ["研究背景", "研究意义"]:
                temp_pred, temp_ref = pred[field], reference[field]
            elif field in ["研究问题", "研究方法", "研究目标"]:
                temp_pred, temp_ref = pred["研究内容"][field], reference["研究内容"][field]
            if temp_pred == "" or temp_ref == "":
                if temp_pred.strip() == temp_ref.strip():
                    scores = [{'rouge-1': {'r': 1.0, 'p': 1.0, 'f': 1.0}, 'rouge-2': {'r': 1.0, 'p': 1.0, 'f': 1.0}, 'rouge-l': {'r': 1.0, 'p': 1.0, 'f': 1.0}}]
                else:
                    scores = [{'rouge-1': {'r': 0.0, 'p': 0.0, 'f': 0.0}, 'rouge-2': {'r': 0.0, 'p': 0.0, 'f': 0.0}, 'rouge-l': {'r': 0.0, 'p': 0.0, 'f': 0.0}}]
            else:
                temp_pred = ' '.join(jieba.cut(str(temp_pred)))
                temp_ref = ' '.join(jieba.cut(str(temp_ref)))
                print([temp_pred], [temp_ref])
                scores = rouge.get_scores([temp_pred], [temp_ref])
            results[field].append(scores[0])
    for field in field_list:
        results_avg[field] = {}
        for metric in metrics_list:
            results_avg[field][metric] = {}
            for prf in ["p", "r", "f"]:
                results_avg[field][metric][prf] = np.mean([record[metric][prf] for record in results[field]])
    # bertscore
    results = {field: [] for field in field_list}
    test_data = read_jsonl_by_line(path_test_jsonl)
    for index, record in enumerate(test_data):
        if record["conversation_id"] <= 180:
            continue
        # 数据读入
        reference = json.loads(record["conversation"][0]["assistant"])
        pred = json.loads(record["conversation"][0]["pred"])
        pred = result_format_check(pred)
        # 分字段评价
        for field in field_list:
            if field in ["研究背景", "研究意义"]:
                temp_pred, temp_ref = pred[field], reference[field]
            elif field in ["研究问题", "研究方法", "研究目标"]:
                temp_pred, temp_ref = pred["研究内容"][field], reference["研究内容"][field]
            scores = bert_score_evaluate([str(temp_ref)], [str(temp_pred)])
            results[field].append(scores)
    metric = "bert-score"
    for field in field_list:
        # results_avg[field] = {}
        results_avg[field][metric] = {}
        for prf in ["p", "r", "f"]:
            results_avg[field][metric][prf] = np.mean([record[prf] for record in results[field]])
    print(results_avg)
    return results_avg


if __name__ == '__main__':
    print('evaluate.py')
    # # 整体计算-推理：0样本和1样本
    # save_dic = {}
    # for example in ["zero", "one"]:
    #     save_dic[example] = {}
    #     for model in ["llama2_7b_chat", "llama2_13b_chat", "llama2_70b_chat", "llama3_8b_chat", "llama3_70b_chat", "mistral-7b-v0.1", "mixtral_8x7b-v0.1", "qwen_14b_chat", "qwen_72b_chat"]:
    #         path_jsonl = f"/home/llmtrainer/yf_project/research_question_tree_jy/data/dataset0804/test_for_paper_data/{example}_shot_test_40_predict_{model}.jsonl"
    #         result = rouge_and_bertscore_as_whole(path_jsonl)
    #         print(example, model, result)
    #         save_dic[example][model] = result
    # print("="*10, "\n\n\n")
    # print(save_dic)
    # logger.info(save_dic)
    # # 整体计算-微调
    # save_dic = {}
    # for model in ["llama-7b", "llama-13b", "mistral-7b", "mixtral-8x7b", "qwen-14b"]:
    #     path_jsonl = f"/home/llmtrainer/yf_project/research_question_tree_jy/data/dataset0804/simple/test_predict_{model}-simple_1800.jsonl"
    #     result = rouge_and_bertscore_as_whole(path_jsonl)
    #     save_dic[model] = result
    # print("="*10, "\n\n\n")
    # print(save_dic)

    # 分字段计算-微调
    # save_dic = {}
    # for model in ["llama-7b", "llama-13b", "mistral-7b", "mixtral-8x7b", "qwen-14b"]:
    #     path_jsonl = f"/home/llmtrainer/yf_project/research_question_tree_jy/data/dataset0804/simple/test_predict_{model}-simple_1800.jsonl"
    #     result = rouge_and_bertscore_each_field(path_jsonl)
    #     save_dic[model] = result
    # print("="*10, "\n\n\n")
    # print(save_dic)


    # 微调
    # text_rouge_score("data/dataset0804/test_for_paper_data/one_shot_test_40_predict_llama3_70b_chat.jsonl")
    # get_evaluate_value("/home/llmtrainer/yf_project/research_question_tree_jy/data/dataset0804/test_for_paper_data/one_shot_test_40_predict_llama2_7b_chat.jsonl",
    #                    "rouge"  #    "bertscore", "rouge"
    #                    )
    evaluate_prf_v3("/data01/public/yifan/grant_match_v3/data_match/0919/qwen3-0.6b_human_test.jsonl")
    print("\n", "="*12, "\n")
    evaluate_prf_v3("/data01/public/yifan/grant_match_v3/data_match/0919/qwen3-4b_human_test.jsonl")
    print("\n", "="*12, "\n")
    evaluate_prf_v3("/data01/public/yifan/grant_match_v3/data_match/0919/qwen3-8b_human_test.jsonl")
    # evaluate_rouge("/home/llmtrainer/yf_project/research_question_tree_jy/data/data0823/test_predict_llama2-13b_1800.jsonl")
'''
该程序用于将MedQA数据集MedQA_chinese_qbank.jsonl转换为alpaca格式，输出为MedQA_chinese_qbank_alpaca.json
'''

import json

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))  # 每行数据转换为字典并加入到列表中
    return data

def convert_to_alpaca_format(data):
    alpaca_data = []

    for item in data:
        question = item["question"]
        options = item["options"]
        answer = item["answer"]
        meta_info = item["meta_info"]

        alpaca_data.append({
            "instruction": f"你是一个{meta_info}问题助手，帮助用户解答{meta_info}问题。请根据问题选择正确的答案，并只输出字母（A、B、C、D或E）。",
            "input": f"{question}\nA. {options[0]}\nB. {options[1]}\nC. {options[2]}\nD. {options[3]}\nE. {options[4]}",
            "output": f"{answer}"
        })

    return alpaca_data

data = read_jsonl("./train/data/MedQA_chinese_qbank.jsonl")
alpaca_data = convert_to_alpaca_format(data)

# 输出到 JSON 文件
output_file_path = './train/data/MedQA_chinese_qbank_alpaca.json'  # 修改为你想要保存的文件路径
with open(output_file_path, "w", encoding="utf-8") as f:
    json.dump(alpaca_data, f, ensure_ascii=False, indent=2)

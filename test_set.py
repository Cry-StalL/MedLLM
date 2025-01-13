import csv
import re
import os
from datetime import datetime
from utils.llm_handler import LLMHandler
from utils.config_loader import ConfigLoader
from utils.utils import Extract_answer

no_response_count = 0

def load_test_set(filename):
    data = []
    with open(filename, newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            question_data = {
                'question': row['question'],
                'answerA': row['answerA'],
                'answerB': row['answerB'],
                'answerC': row['answerC'],
                'answerD': row['answerD'],
                'answerE': row['answerE'],
            }
            data.append(question_data)
    return data


config = ConfigLoader().get_config()

test_set = load_test_set('./data/test_set.csv')  # 加载测试集

model_instance = LLMHandler(config['model']['name_or_path'])

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_folder = "./output"
output_path = "./output/test_set_" + current_time + ".csv"  # 输出为测试集结果

num = 0

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

with open(output_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['id', 'question', 'answerA', 'answerB', 'answerC', 'answerD', 'answerE', 'modelResponse', 'modelAnswer'])  # 只有模型响应和模型答案

    for question in test_set:
        prompt = (f"请根据问题选择正确的答案，并只输出字母（A、B、C、D或E）：\n"
                  f"问题：{question['question']}\n\n"
                  f"A.{question['answerA']}\n"
                  f"B.{question['answerB']}\n"
                  f"C.{question['answerC']}\n"
                  f"D.{question['answerD']}\n"
                  f"E.{question['answerE']}\n\n")

        method = config['model']['method']
        if method == 'direct_prompt':
            response = model_instance.generate_response_in_a_letter(prompt)
        elif method == 'cot':
            response = model_instance.query_response_cot(question)
        elif method == 'scot':
            response = model_instance.query_response_scot(question)
        elif method == 'knowledge_prompt':
            response = model_instance.query_response_with_knowledge_prompt(question)
        elif method == 'few_shot_prompt':
            response = model_instance.query_response_with_few_shot_prompt(question)
        else:
            raise ValueError('Unknown method set in config.yml')

        extracted_answer = Extract_answer(response, no_response_count)

        num = num + 1

        # 将数据写入 CSV 文件
        writer.writerow([num,
                         question['question'],
                         question['answerA'],
                         question['answerB'],
                         question['answerC'],
                         question['answerD'],
                         question['answerE'],
                         response,
                         extracted_answer,
                         ])
        if(num % 10 == 0):
            file.flush()  # 每10条数据保存一次文件

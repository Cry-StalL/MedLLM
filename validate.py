import csv
import re
import os
from datetime import datetime
from utils.llm_handler import LLMHandler
from utils.config_loader import ConfigLoader
from utils.utils import Extract_answer

no_response_count = 0

def load_validation_set(filename):
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
                'correctAnswer': row['correctAnswer']
            }
            data.append(question_data)
    return data


config = ConfigLoader().get_config()

validation_set = load_validation_set('./data/validation_set.csv')

model_instance = LLMHandler(config['model']['name_or_path'])

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_folder = "./output"
output_path = "./output/" + current_time + ".csv"

num = 0
correct_num = 0

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

with open(output_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['id', 'question', 'answerA', 'answerB', 'answerC', 'answerD', 'answerE', 'correctAnswer','modelResponse', 'modelAnswer', 'is_correct'])

    for question in validation_set:
        prompt = (f"请根据问题选择正确的答案，并只输出字母（A、B、C、D或E）：\n"
                  f"问题：{question['question']}\n\n"
                  f"A.{question['answerA']}\n"
                  f"B.{question['answerB']}\n"
                  f"C.{question['answerC']}\n"
                  f"D.{question['answerD']}\n"
                  f"E.{question['answerE']}\n\n")

        # print(f"prompt:\n{prompt}\n")

        response = model_instance.generate_response_in_a_letter(prompt)

        extracted_answer = Extract_answer(response, no_response_count)

        num = num + 1
        if extracted_answer == question['correctAnswer']:
            correct_num = correct_num + 1

        # print(f"response:{response}\nanswer:{extracted_answer}")
        if(num%10==0):
            print(f"目前正确率:{correct_num}/{num}={correct_num / num:.2f}\n") #每10条数据输出一次

        # 将数据写入 CSV 文件
        writer.writerow([num,
                         question['question'],
                         question['answerA'],
                         question['answerB'],
                         question['answerC'],
                         question['answerD'],
                         question['answerE'],
                         question['correctAnswer'],
                         response,
                         extracted_answer,
                         1 if extracted_answer == question['correctAnswer'] else 0,
                         ])
        if(num%10==0):
            file.flush() #每10条数据保存一次文件
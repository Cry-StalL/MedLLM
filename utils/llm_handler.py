import random
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from prompts import knowledge_prompt, few_shot_prompts
from utils import Extract_closest_tail_uppercase_letter
from utils.config_loader import ConfigLoader


class LLMHandler:
    def __init__(self, model_name: str):
        self.config = ConfigLoader().get_config()
        self.model_name = model_name  # Qwen/Qwen2.5-Coder-0.5B 或 Qwen/Qwen2.5-Coder-0.5B-Instruct
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        # 自动检测可用的设备 (GPU / CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 加载模型并将其移动到合适的设备
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        # 输出模型当前所在的设备
        print(f"Model is loaded on: {self.model.device}")

    # 通用函数，简单的query，不含思考过程，通过messages list获取response
    def query_with_messages(self, messages: list) -> str:
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.config['model']['max_token'],
            do_sample=self.config['model']['do_sample'],
            temperature=self.config['model']['temperature'],
            pad_token_id=self.pad_token_id,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response

    # 通用函数，复杂的query，有思考过程
    def query_with_messages_complex(self, messages: list) -> str:
        # 最多重试
        max_retries = 3

        for i in range(max_retries):

            answer = self.query_with_messages(messages)
            # 假设！！答案是最接近尾部的英文大写字母
            # 获取最靠近尾部的第一个英文大写字母
            parsed_answer = Extract_closest_tail_uppercase_letter(answer)

            if parsed_answer is None:
                # 如果没有找到答案，则重新提问
                continue

            # 成功
            return parsed_answer

        # 走到这说明没有answer，优雅地终止程序
        raise RuntimeError("未能获取答案，在尝试多次后依然失败，请检查问题或系统逻辑。")

    # 直接生成一个字母作为答案
    def generate_response_in_a_letter(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        return self.query_with_messages(messages)

    # 使用CoT思维链方式获取答案，不知道这个模型有没有这个能力，能不能成功
    def query_response_cot(self, question: dict) -> str:

        identity = "你是一名资深的医学教授，精通现代医学的各个领域，包括但不限于中医学和西医学。你能够以专业的医学知识解答医学选择题。"

        question = (f"问题：{question['question']}\n\n"
                    f"A.{question['answerA']}\n"
                    f"B.{question['answerB']}\n"
                    f"C.{question['answerC']}\n"
                    f"D.{question['answerD']}\n"
                    f"E.{question['answerE']}\n\n")

        engage_in_cot = "请逐步分析和推理每个选项的合理性与适用性，在**详细思考**后给出最终答案，并以字母形式明确标注，不要在开头给出答案，答案只能在结尾提供。"

        messages = [
            {"role": "system", "content": identity},
            {"role": "user", "content": question + "\n" + engage_in_cot},
        ]

        try:
            answer = self.query_with_messages_complex(messages)
        except Exception as e:
            print(e)
            answer = "无法获取答案"

        return answer

    # Self-consistency CoT或者自平衡思维链
    def query_response_scot(self, question: dict) -> str:
        # 设定平衡次数为5
        balance_times = 5

        answer_list = []

        for i in range(balance_times):
            answer_list.append(self.query_response_cot(question))

        # 多数投票
        # 统计每个答案出现的次数
        answer_counts = Counter(answer_list)

        # 找出出现次数最多的答案
        most_common_answer, _ = answer_counts.most_common(1)[0]

        return most_common_answer

    # 提供领域知识
    def query_response_with_knowledge_prompt(self, question: dict) -> str:

        question = (f"问题：{question['question']}\n\n"
                    f"A.{question['answerA']}\n"
                    f"B.{question['answerB']}\n"
                    f"C.{question['answerC']}\n"
                    f"D.{question['answerD']}\n"
                    f"E.{question['answerE']}\n\n")

        engage_in_cot = "请逐步分析和推理每个选项的合理性与适用性，在**详细思考**后给出最终答案，并以字母形式明确标注，不要在开头给出答案，答案只能在结尾提供。"

        messages = [
            {"role": "system", "content": knowledge_prompt},
            {"role": "user", "content": question + "\n" + engage_in_cot},
        ]

        try:
            answer = self.query_with_messages_complex(messages)
        except Exception as e:
            print(e)
            answer = "无法获取答案"

        return answer

    # 提供few-shot示例
    def query_response_with_few_shot_prompt(self, question: dict) -> str:

        # few-shot数量
        few_shot_num = 4

        identity = "你是一名资深的医学教授，精通现代医学的各个领域，包括但不限于中医学和西医学。你能够以专业的医学知识解答医学选择题。"

        question = (f"问题：{question['question']}\n\n"
                    f"A.{question['answerA']}\n"
                    f"B.{question['answerB']}\n"
                    f"C.{question['answerC']}\n"
                    f"D.{question['answerD']}\n"
                    f"E.{question['answerE']}\n\n")

        engage_in_cot = "请逐步分析和推理每个选项的合理性与适用性，在**详细思考**后给出最终答案，并以字母形式明确标注，不要在开头给出答案，答案只能在结尾提供。"

        # 打乱 few-shot 示例顺序
        random.shuffle(few_shot_prompts)

        # 构建消息列表
        messages = [{"role": "system", "content": identity}]
        for example in few_shot_prompts[:few_shot_num]:
            messages.extend(example)

        # 原本要问的问题
        messages.append({"role": "user", "content": question + "\n" + engage_in_cot})

        try:
            answer = self.query_with_messages_complex(messages)
        except Exception as e:
            print(e)
            answer = "无法获取答案"

        return answer

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.ConfigLoader import ConfigLoader

class LLMHandler:
    def __init__(self, model_name: str):
        self.config = ConfigLoader().get_config()
        self.model_name = model_name # Qwen/Qwen2.5-Coder-0.5B 或 Qwen/Qwen2.5-Coder-0.5B-Instruct
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # 自动检测可用的设备 (GPU / CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 加载模型并将其移动到合适的设备
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        # 输出模型当前所在的设备
        print(f"Model is loaded on: {self.model.device}")

    def generate_response(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.config['model']['max_token'],
            temperature=self.config['model']['temperature']
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response

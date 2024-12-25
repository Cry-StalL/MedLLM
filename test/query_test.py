import sys
sys.path.append('.')

from llm_handler import LLMHandler

if __name__ == '__main__':
    model_instance = LLMHandler("Qwen/Qwen2.5-Coder-0.5B")
    response = model_instance.generate_response("Who are you?")
    print(response)
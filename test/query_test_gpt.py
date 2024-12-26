import sys
sys.path.append('.')

from utils.llm_handler import GPTHandler

if __name__ == '__main__':
    model_instance = GPTHandler("gpt-3.5-turbo")
    response = model_instance.generate_response("Who are you?")
    print(response)
import sys
sys.path.append('.')

from utils.llm_handler import LLMHandler
from utils.config_loader import ConfigLoader

if __name__ == '__main__':
    config = ConfigLoader().get_config()

    model_instance = LLMHandler(config['model']['name_or_path'])
    response = model_instance.generate_response_in_a_letter("你是谁?")
    print(response)
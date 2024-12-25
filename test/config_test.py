import sys
sys.path.append('.')

from utils.config_loader import ConfigLoader

config = ConfigLoader().get_config()

print(config)
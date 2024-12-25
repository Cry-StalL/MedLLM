import sys
sys.path.append('.')

from utils.ConfigLoader import ConfigLoader

config = ConfigLoader().get_config()

print(config)
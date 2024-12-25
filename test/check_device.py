import torch

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 输出当前使用的设备
print(f"Using device: {device}")

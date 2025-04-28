import torch
import torch.nn as nn
import torch.optim as optim

# 檢查 GPU 是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

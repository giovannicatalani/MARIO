import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device(2 if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')
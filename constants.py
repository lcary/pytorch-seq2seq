import torch

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch.nn as nn

def criterion(out, label):
    return nn.functional.cross_entropy(out, label)
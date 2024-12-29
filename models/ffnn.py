# region imports
from AlgorithmImports import *
# endregion
import torch
import torch.nn as nn  
import torch.nn.functional as F

class FFNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(15, 40, bias=True),
            nn.LayerNorm(40),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(40, 16, bias=True),
            nn.LayerNorm(16),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(16, 3, bias=True),
        )

    def forward(self, x):
        return F.softmax(self.net(x), dim=1)

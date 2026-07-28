import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        self.embedding = nn.Embedding(vocabulary_size, 16)
        self.linear = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
        pass

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        embeds = self.embedding(x)
        averaged = embeds.mean(dim=1)
        logits = self.linear(averaged)
        probs = self.sigmoid(logits)

        return torch.round(probs * 10000) / 10000
       

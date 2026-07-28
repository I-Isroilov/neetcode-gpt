import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.key = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value = nn.Linear(embedding_dim, attention_dim, bias=False)


    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        B, T, _ = embedded.shape
        
        k = self.key(embedded)
        q = self.query(embedded)
        v = self.value(embedded)

        attention_dim = k.shape[-1]
        scores = q @ k.transpose(1, 2) / (attention_dim ** 0.5)

        mask = torch.tril(torch.ones(T,T))
        scores = scores.masked_fill(mask == 0, float('-inf'))

        scores = torch.softmax(scores, dim=2)

        out = scores @ v

        return torch.round(out * 10000) / 10000



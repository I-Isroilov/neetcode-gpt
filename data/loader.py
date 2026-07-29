import torch
from torchtyping import TensorType
from typing import Tuple

class Solution:
    def create_batches(self, data: TensorType[int], context_length: int, batch_size: int) -> Tuple[TensorType[int], TensorType[int]]:
        torch.manual_seed(0)
        high = len(data) - context_length
        starts = torch.randint(low=0, high=high, size=(batch_size,))

        X = torch.stack([data[start: start + context_length] for start in starts])
        Y = torch.stack([data[start + 1: start + context_length + 1] for start in starts])

        return X, Y
        

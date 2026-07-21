import torch
import math
from typing import List


class Solution:

    def _std(self, init_type: str, fan_in: int, fan_out: int) -> float:
        if init_type == 'xavier':
            return math.sqrt(2.0 / (fan_in + fan_out))
        elif init_type == 'kaiming':
            return math.sqrt(2.0 / fan_in)
        elif init_type == 'random':
            return 1.0
        raise ValueError(f"unknown init type: {init_type}")

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        std = math.sqrt(2.0 / (fan_in + fan_out))
        w = torch.randn(fan_out, fan_in) * std
        return [[round(v, 4) for v in row] for row in w.tolist()]

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        std = math.sqrt(2.0 / fan_in)
        w = torch.randn(fan_out, fan_in) * std
        return [[round(v, 4) for v in row] for row in w.tolist()]

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        torch.manual_seed(0)
        # 1) build all weight matrices first (one continuous RNG stream)
        weights = []
        in_dim = input_dim
        for _ in range(num_layers):
            std = self._std(init_type, in_dim, hidden_dim)
            weights.append(torch.randn(hidden_dim, in_dim) * std)
            in_dim = hidden_dim
        # 2) then draw the input vector and forward through the stack
        x = torch.randn(input_dim)
        stds = []
        for w in weights:
            x = torch.relu(w @ x)
            stds.append(round(x.std().item(), 2))
        return stds
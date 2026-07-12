import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        std = math.sqrt(2.0 / (fan_in + fan_out))
        w = torch.randn(fan_out, fan_in) * std
        return [[round(v.item(), 4) for v in row] for row in w]

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        std = math.sqrt(2.0 / fan_in)
        w = torch.randn(fan_out, fan_in) * std
        return [[round(v.item(), 4) for v in row] for row in w]

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        torch.manual_seed(0)

        dims = [input_dim] + [hidden_dim] * num_layers
        weights = []

        # Important: generate all weights first, then the input vector.
        for i in range(num_layers):
            if init_type == "kaiming":
                std = math.sqrt(2.0 / dims[i])
            elif init_type == "xavier":
                std = math.sqrt(2.0 / (dims[i] + dims[i + 1]))
            else:  # random
                std = 1.0

            w = torch.randn(dims[i + 1], dims[i]) * std
            weights.append(w)

        x = torch.randn(1, input_dim)
        stds = []

        for w in weights:
            x = x @ w.T
            x = torch.relu(x)
            stds.append(round(x.std().item(), 2))

        return stds
import math
import torch
import torch.nn as nn
from torchtyping import TensorType


class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)

        # Order matters: key, query, value
        self.key = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value = nn.Linear(embedding_dim, attention_dim, bias=False)

        self.attention_dim = attention_dim

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # embedded: (B, T, embedding_dim)

        K = self.key(embedded)      # (B, T, attention_dim)
        Q = self.query(embedded)    # (B, T, attention_dim)
        V = self.value(embedded)    # (B, T, attention_dim)

        # (B, T, T)
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / math.sqrt(self.attention_dim)

        # Causal mask
        T = embedded.size(1)
        mask = torch.tril(torch.ones(T, T, device=embedded.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))

        # Attention probabilities
        scores = torch.softmax(scores, dim=2)

        # (B, T, attention_dim)
        output = torch.matmul(scores, V)

        return torch.round(output, decimals=4)
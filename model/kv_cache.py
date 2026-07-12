import torch
import torch.nn as nn
from typing import Tuple, Optional

class KVCache:
    def __init__(self):
        self.cache_k: Optional[torch.Tensor] = None  # (batch, seq_len, model_dim)
        self.cache_v: Optional[torch.Tensor] = None

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Initialize cache if it's the first token(s) being processed
        if self.cache_k is None or self.cache_v is None:
            self.cache_k = new_k
            self.cache_v = new_v
        else:
            # Append new keys and values along the sequence dimension (dim=1)
            self.cache_k = torch.cat([self.cache_k, new_k], dim=1)
            self.cache_v = torch.cat([self.cache_v, new_v], dim=1)
            
        return self.cache_k, self.cache_v

    def clear(self):
        self.cache_k = None
        self.cache_v = None

class CachedAttention(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None) -> Tuple[torch.Tensor, KVCache]:
        # 1. Project x into Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. If kv_cache is None, create a new KVCache
        if kv_cache is None:
            kv_cache = KVCache()

        # 3. Update the cache with the new K and V
        full_k, full_v = kv_cache.update(k, v)

        # 4. Compute scaled dot-product attention
        d_k = q.shape[-1]
        
        # Q: (batch, current_seq_len, model_dim)
        # full_K: (batch, full_seq_len, model_dim)
        # Transpose full_K to (batch, model_dim, full_seq_len) for matrix multiplication
        scores = torch.matmul(q, full_k.transpose(1, 2)) / (d_k ** 0.5)
        
        # Apply softmax over the full sequence length dimension
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)
        
        # Multiply weights by full_V: (batch, full_seq_len, model_dim)
        output = torch.matmul(attention_weights, full_v)

        # 5. Return (rounded output, kv_cache)
        return torch.round(output, decimals=4), kv_cache
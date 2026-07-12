import torch
from torchtyping import TensorType
from typing import Tuple


class Solution:
    def create_batches(
        self,
        data: TensorType[int],
        context_length: int,
        batch_size: int
    ) -> Tuple[TensorType[int], TensorType[int]]:

        torch.manual_seed(0)

        # Valid start positions
        starts = torch.randint(
            0,
            len(data) - context_length,
            (batch_size,)
        )

        X = torch.stack([
            data[s:s + context_length]
            for s in starts
        ])

        Y = torch.stack([
            data[s + 1:s + context_length + 1]
            for s in starts
        ])

        return X, Y
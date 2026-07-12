import torch
from typing import List, Tuple


class Solution:
    def batch_loader(
        self,
        raw_dataset: str,
        context_length: int,
        batch_size: int
    ) -> Tuple[List[List[str]], List[List[str]]]:

        # 1. Tokenize
        tokens = raw_dataset.split()

        # 2. Random start indices
        torch.manual_seed(0)
        starts = torch.randint(
            low=0,
            high=len(tokens) - context_length,
            size=(batch_size,)
        )

        X = []
        Y = []

        # 3. Create input/target sequences
        for start in starts.tolist():
            X.append(tokens[start:start + context_length])
            Y.append(tokens[start + 1:start + 1 + context_length])

        return X, Y
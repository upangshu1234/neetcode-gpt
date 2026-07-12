import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List


class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # Build vocabulary
        vocab = sorted(set(
            word
            for sentence in positive + negative
            for word in sentence.split()
        ))
        word_to_id = {word: i + 1 for i, word in enumerate(vocab)}

        # Encode sentences
        tensors = []
        for sentence in positive + negative:
            ids = [word_to_id[word] for word in sentence.split()]
            tensors.append(torch.tensor(ids, dtype=torch.float32))

        # Pad sequences
        dataset = nn.utils.rnn.pad_sequence(
            tensors,
            batch_first=True,
            padding_value=0
        )

        return dataset
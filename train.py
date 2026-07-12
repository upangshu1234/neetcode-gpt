import torch
import torch.nn as nn
import torch.nn.functional as F


class Solution:
    def train(
        self,
        model: nn.Module,
        data: torch.Tensor,
        epochs: int,
        context_length: int,
        batch_size: int,
        lr: float
    ) -> float:

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        loss = None

        for epoch in range(epochs):
            torch.manual_seed(epoch)

            # Sample random starting positions
            starts = torch.randint(
                0,
                len(data) - context_length,
                (batch_size,)
            )

            # Build input/target batches
            X = torch.stack(
                [data[s:s + context_length] for s in starts]
            )

            Y = torch.stack(
                [data[s + 1:s + context_length + 1] for s in starts]
            )

            optimizer.zero_grad()

            # Forward pass (model returns raw logits)
            logits = model(X)

            # Cross-entropy over all tokens
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                Y.view(-1)
            )

            loss.backward()
            optimizer.step()

        return round(loss.item(), 4)
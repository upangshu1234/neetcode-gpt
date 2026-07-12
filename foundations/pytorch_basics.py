import torch
import torch.nn.functional as F
from torchtyping import TensorType


class Solution:
    def reshape(self, to_reshape: TensorType[float]) -> TensorType[float]:
        # Reshape (M, N) -> (M*N/2, 2)
        return torch.round(
            torch.reshape(to_reshape, (-1, 2)),
            decimals=4
        )

    def average(self, to_avg: TensorType[float]) -> TensorType[float]:
        # Column-wise mean
        return torch.round(
            torch.mean(to_avg, dim=0),
            decimals=4
        )

    def concatenate(
        self,
        cat_one: TensorType[float],
        cat_two: TensorType[float]
    ) -> TensorType[float]:
        # Concatenate side-by-side
        return torch.round(
            torch.cat((cat_one, cat_two), dim=1),
            decimals=4
        )

    def get_loss(
        self,
        prediction: TensorType[float],
        target: TensorType[float]
    ) -> TensorType[float]:
        # Mean Squared Error
        return torch.round(
            F.mse_loss(prediction, target),
            decimals=4
        )
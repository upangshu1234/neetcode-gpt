import numpy as np
from numpy.typing import NDArray


class Solution:

    def get_model_prediction(
        self,
        X: NDArray[np.float64],
        weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # X: (n, m), weights: (m,)
        predictions = X @ weights
        return np.round(predictions, 5)

    def get_error(
        self,
        model_prediction: NDArray[np.float64],
        ground_truth: NDArray[np.float64]
    ) -> float:
        mse = np.mean((model_prediction - ground_truth) ** 2)
        return round(mse, 5)
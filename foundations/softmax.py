import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # Numerical stability
        z = z - np.max(z)

        exp_z = np.exp(z)
        softmax = exp_z / np.sum(exp_z)

        return np.round(softmax, 4)
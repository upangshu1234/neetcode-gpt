import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        # Position indices: (seq_len, 1)
        position = np.arange(seq_len)[:, np.newaxis]

        # Even dimension indices: (d_model/2,)
        div_term = np.exp(
            np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        )

        pe = np.zeros((seq_len, d_model), dtype=np.float64)

        # Even columns
        pe[:, 0::2] = np.sin(position * div_term)

        # Odd columns
        pe[:, 1::2] = np.cos(position * div_term)

        return np.round(pe, 5)
import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        position = np.arange(seq_len)[:, np.newaxis]
        i = np.arange(0, d_model, 2)

        div_term = np.power(10000.0, i / d_model)

        angles = position / div_term

        PE = np.zeros((seq_len, d_model))
        PE[:, 0::2] = np.sin(angles)
        PE[:, 1::2] = np.cos(angles)

        return np.round(PE, 5)



        

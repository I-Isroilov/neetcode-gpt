import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        sigmoid = 1 / (1 + np.exp(-z))
        return np.round(sigmoid, 5)

        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        pass

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.maximum(0, z)
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        pass

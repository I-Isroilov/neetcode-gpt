import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        z = float(np.dot(x,w) + b)
        sigmoid = 1/(1 + np.exp(-z))
        y_hat = sigmoid
        
        dL_dyhat = (y_hat - y_true)              
        dyhat_dz = sigmoid * (1 - sigmoid)        
        dL_dz = dL_dyhat * dyhat_dz               

        dL_dw = dL_dz * x                         
        dL_db = dL_dz                             

        return np.round(dL_dw, 5), round(float(dL_db), 5)
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        

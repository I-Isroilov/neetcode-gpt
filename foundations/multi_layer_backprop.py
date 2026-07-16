import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        
        n_in = len(x)
        n_hidden = len(W1)
        n_out = len(W2)
        m = len(y_true)

        #-------forward pass-------
        z1 = [sum(W1[i][j] * x[j] for j in range(n_in)) + b1[i] for i in range(n_hidden)]
        a1 = [z if z > 0 else 0.0 for z in z1]
        z2 = [sum(W2[k][i] * a1[i] for i in range(n_hidden)) + b2[k] for k in range(n_out)]
        prediction = z2

        loss = sum((prediction[k] - y_true[k]) ** 2 for k in range(m)) / m

        #--------backward pass------
        dz2 = [2.0 * (prediction[k] - y_true[k]) / m for k in range(n_out)]
        dW2 = [[dz2[k] * a1[i] for i in range(n_hidden)] for k in range(n_out)]

        db2 = [dz2[k] for k in range(n_out)]

        da1 = [sum(dz2[k] * W2[k][i] for k in range(n_out)) for i in range(n_hidden)]

        dz1 = [da1[i] * (1.0 if z1[i] > 0 else 0.0) for i in range(n_hidden)]
        dW1 = [[dz1[i] * x[j] for j in range(n_in)] for i in range(n_hidden)]
        db1 = [dz1[i] for i in range(n_hidden)]


        return {
            'loss': round(loss, 4),
            'dW1': [[round(v, 4) for v in row] for row in dW1],
            'db1': [round(v, 4) for v in db1],
            'dW2':[[round(v, 4) for v in row] for row in dW2],
            'db2': [round(v, 4) for v in db2] ,
        }

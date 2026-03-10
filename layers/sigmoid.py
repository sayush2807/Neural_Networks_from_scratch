import numpy as np

class sigmoid:
    def __init__(self,z):
        self.y_hat = None

    def forward(self, z):
        self.y_hat = 1/(1+np.exp(-z))
        return y_hat
    
    def backward(self, dz):
        # calculates its on part of chain rule, dy_hat/dz = x
        
        return dz * self.y_hat*

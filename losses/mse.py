import numpy as np
def mse_loss(y, y_hat):
    return np.mean((y-y_hat)**2)


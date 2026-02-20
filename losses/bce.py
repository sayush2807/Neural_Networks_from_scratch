import numpy as np
# binary cross entropy loss for logistic regression
def bce_loss(y_hat,y):
    return np.mean(-y*np.log(y)-(1-y_hat)*np.log(1-y_hat))
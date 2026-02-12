from losses.mse import mse_loss
from optimizer.SGD import SGD
import numpy as np

def train(linear, x,y, epochs, lr):
    optimizer = SGD(params = linear.parameters(), lr = lr )
    for i in range(epochs):
        y_hat = linear.forward(x)
        print("Epoch ", i+1, " : MSE Loss", mse_loss(y,y_hat))
        linear.backward(y,y_hat)
        linear.backward(y, y_hat)
        #gradient norm check -- these should be non-zero and reasonable
        print("||dW|| =", np.linalg.norm(linear.gradient_w))
        print("||db|| =", np.linalg.norm(linear.gradient_b))
        optimizer.step()

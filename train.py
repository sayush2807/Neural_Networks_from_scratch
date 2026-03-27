from losses.mse import bce_loss
from optimizer.SGD import SGD
import numpy as np
from models import LogisticRegression
# train logistic regression

def train(linear, x,y, epochs, lr, loss):
    logistic_regressor = LogisticRegression()
    optimizer = SGD(params = linear.parameters(), lr = lr )
    sigmoid = Sigmoid()
    for i in range(epochs):
        y_hat = linear.forward(x)
        print("Epoch ", i+1, " : Loss", loss(y,y_hat))
        linear.backward(y,y_hat)
        
        #gradient norm check -- these should be non-zero and reasonable
        print("||dW|| =", np.linalg.norm(linear.gradient_w))
        print("||db|| =", np.linalg.norm(linear.gradient_b))
        optimizer.step()

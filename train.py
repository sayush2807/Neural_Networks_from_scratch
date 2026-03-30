from losses.bce import bce_loss
from optimizer.SGD import SGD
import numpy as np
from models.logistic_regression import LogisticRegression
# train logistic regression

#create data

def train(x,y, epochs, lr, loss):
    model = LogisticRegression(n_features = x.shape[1])
    loss_fn = BCELoss()
    optimizer = SGD(params = linear.parameters(), lr = lr )
    for i in range(epochs):
        y_hat = model.forward(x)
        loss = loss_fn.forward(y,y_hat)
        da = loss_fn.backward(y,y_hat)
        model.backward(da)
        optimizer.step()
        print("Epoch ", i+1, " : Loss", loss)
        #gradient norm check -- these should be non-zero and reasonable
        print("||dW|| =", np.linalg.norm(linear.gradient_w))
        print("||db|| =", np.linalg.norm(linear.gradient_b))

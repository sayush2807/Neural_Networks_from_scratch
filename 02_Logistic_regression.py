import numpy as np
from layers.linear import Dense
from train import train 
from losses.bce import bce_loss


# create input for logistic regression
n_samples = 100
n_features = 10
x = np.random.rand(n_samples, n_features)

# create output 
w_true = np.random.rand(n_features,1)
b_true = np.random.rand(1)
noise = 0.05 * np.random.randn(n_samples, 1)
y_without_sigmoid = x @ w_true+b_true+noise
y = 1/(1+np.exp(-y_without_sigmoid))
y = [1 if i>=0.5 else 0 for i in y]

linear = Dense(n_features, 1)

train(linear, x,y,lr=0.01, bce_loss)


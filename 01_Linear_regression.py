import numpy as np
from layers.linear import Dense # folder -->python file -->class
from train import train

# create input 
n_samples = 10
n_features = 5
x = np.random.rand(n_samples, n_features) # 10 samples, 5 features

# create output 
w_true = np.random.randn(n_features, 1)
b_true = np.random.randn(1)
noise = 0.05 * np.random.randn(n_samples, 1)
y = x @ w_true + b_true + noise

linear = Dense(n_features, 1)

train(linear, x, y, epochs = 50, lr = 0.1)
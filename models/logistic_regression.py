from layers.linear import Dense
from layers.sigmoid import Sigmoid


class LogisticRegressionClass:
    def __init__(self, n_features):
        self.linear = Dense(n_features, 1)
        self.sigmoid = Sigmoid()
        

    def forward(self, x):
        z = self.linear.forward(x)
        a = self.sigmoid.forward(z)
        return a
        
    def backward(self, da):
        dz = self.sigmoid.backward(da)
        dx = self.linear.backward(dz)
        return dx
    

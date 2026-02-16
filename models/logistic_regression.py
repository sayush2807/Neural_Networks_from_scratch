from layers.linear import Dense, sigmoid
class LogisticRegressionClass:
    def __init__(self, n_features):
        self.linear = Dense(n_features, 1)
        

    def forward(self, x):
        self.linear.forward()

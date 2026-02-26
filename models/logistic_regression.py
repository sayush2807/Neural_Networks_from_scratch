from layers.linear import Dense, sigmoid
from losses import bce
class LogisticRegressionClass:
    def __init__(self, n_features):
        self.linear = Dense(n_features, 1)
        

    def forward(self, x):
        self.linear.forward()
        self.sigmoid.forward()
        self.bce.forward()
        self.bce.backward()# computes gradient of loss wrt y_hat
        self.sigmoid.backward() # this calculates chain rule's next part
        self.linear.backward() #this calculates chain rule'next part
        self.optimizer.step() # think of this , first declare optimizer then how would sigmoid.forward/backward look? will it be optimizer.forward/backward 
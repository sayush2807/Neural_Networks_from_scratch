class SGD:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr
    def step(self):
        for weights, grad in self.params:
            weights -= self.lr*grad
       

            
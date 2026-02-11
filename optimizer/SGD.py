class SGD:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr
    def step(self):
        for p in self.params:
            p.data = p.data - self.lr*p.grad
            
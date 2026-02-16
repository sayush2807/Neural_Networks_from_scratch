import numpy as np

class Dense:
    def __init__(self, input_features, number_of_neurons):
        self.w = np.random.randn(input_features, number_of_neurons)
        self.b = np.zeros((1, number_of_neurons))
        self.x = None
        self.gradient_b = np.zeros_like(self.b)
        self.gradient_w = np.zeros_like(self.w)

    def forward(self, x):
        self.x = x
        return x@self.w+self.b

    #ignore the 2 in gradients as learning rate can be scaled accordingly 
    def backward(self,y, y_hat):
        n_samples = self.x.shape[0]
        dZ = (y_hat-y)/n_samples
        self.gradient_w[:]= ((self.x.T)@dZ)
        gradient_b_temp = sum(dZ)
        # bias is shared across samples, so num gradient of bias across samples and using that in gradient descent
        self.gradient_b[:]= np.sum(gradient_b_temp, axis = 0, keepdims=True)# [:] ensures in place updates of gradient
        # return gradient w.r.t input so previous layers can backprop
        dx = dZ @ self.w.T  # return input gradient (needed for chaining layers)
        return dx

    def parameters(self):
        return [(self.w, self.gradient_w), (self.b, self.gradient_b)]
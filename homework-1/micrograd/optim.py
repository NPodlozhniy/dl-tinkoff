class SGD:
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        """Applying gradient descent to parameters"""
        # Implement SGD!
        for i, parameter in enumerate(self.parameters):
            self.parameters[i].data -= self.lr * self.parameters[i].grad

    def zero_grad(self):
        """Resetting gradient for all parameters (set gradient to zero)"""
        for i, parameter in enumerate(self.parameters):
            self.parameters[i].grad = 0
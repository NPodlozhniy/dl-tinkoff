import numpy as np

from engine import Value, Tensor


class Module:
    """
    Base class for every layer.
    """
    def forward(self, *args, **kwargs):
        """Depends on functionality"""
        pass

    def __call__(self, *args, **kwargs):
        """For convenience we can use model(inp) to call forward pass"""
        return self.forward(*args, **kwargs)

    def parameters(self):
        """Return list of trainable parameters"""
        return []


class Linear(Module):
    def __init__(self, in_features, out_features, bias: bool = True):
        """Initializing model"""
        stdv = 1.0 / np.sqrt(in_features)
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.w = Tensor([Value(np.random.uniform(-stdv, \
                                                 stdv)) for _ in range(self.in_features * \
                                    self.out_features)]).reshape((self.in_features, \
                                                                 self.out_features))
        self.b = Tensor([Value(np.random.uniform(-stdv, stdv)) for _ in range(self.out_features)]) 

    def forward(self, inp):
        """Y = W * x + b"""
        if self.bias:
            return inp.dot(self.w) + self.b
        return inp.dot(self.w)

    def parameters(self):
        if self.bias:
            return self.w.parameters() + self.b.parameters()
        return self.w.parameters()


class ReLU(Module):
    """The most simple and popular activation function"""
    def forward(self, inp):
        # Create ReLU Module
        return Tensor([value.relu() for value in inp.parameters()]).reshape(inp.shape())


class CrossEntropyLoss(Module):
    """
    Cross-entropy loss for multi-class classification
    According to https://pytorch.org/docs/master/generated/torch.nn.CrossEntropyLoss.html
    """
    def forward(self, inp, label):
        # Create CrossEntropy Loss Module
        length = label.shape()[0]
        return - sum([(inp.exp()[i][label[i].data] / sum(inp.exp()[i])).log() for i in range(length)]) / length
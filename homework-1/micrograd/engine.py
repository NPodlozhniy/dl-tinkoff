import numpy as np
from typing import Union

class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op  # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other: Union[int, float, "Value"]) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other: Union[int, float, "Value"]) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other: Union[int, float]) -> "Value":
        assert isinstance(
            other, (int, float)
        ), "only supporting int/float powers for now"
        out = Value(np.power(self.data, other), (self,), f'**{other}')

        def _backward():
            self.grad += other * np.power(self.data, other - 1) * out.grad

        out._backward = _backward

        return out

    def exp(self):
        out = Value(np.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Value(np.log(self.data), (self,), "log")

        def _backward():
            self.grad += out.grad / self.data

        out._backward = _backward

        return out

    def relu(self):
        out = Value((self.data > 0) * self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self):  # -self
        return self * -1

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * other ** -1

    def __rtruediv__(self, other):  # other / self
        return other * self ** -1

    def __le__(self, other):
        if isinstance(other, Value):
            return self.data <= other.data
        return self.data <= other

    def __lt__(self, other):
        if isinstance(other, Value):
            return self.data < other.data
        return self.data < other

    def __gt__(self, other):
        if isinstance(other, Value):
            return self.data > other.data
        return self.data > other

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


from typing import Iterable

class Tensor:
    """
    Tensor is a kinda array with expanded functianality.

    Tensor is very convenient when it comes to matrix multiplication,
    for example in Linear layers.
    """
    def __init__(self, data: Union[Iterable["Value"], Iterable[Iterable["Value"]]]):
        self.data = np.array(data)

    def could_be_broadcasted(self, other):
        return all((m == n) or
                   (m == 1) or
                   (n == 1) for m, n in zip(self.shape()[::-1],
                                            other.shape()[::-1]))

    def __add__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        if isinstance(other, Tensor):
            assert self.could_be_broadcasted(other)
            return Tensor(np.add(self.data, other.data))
        return Tensor(self.data + other)

    def __mul__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        if isinstance(other, Tensor):
            assert self.could_be_broadcasted(other)
            return Tensor(self.data * other.data)
        return Tensor(self.data * other)
    
    def __truediv__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        if isinstance(other, Tensor):
            assert self.could_be_broadcasted(other)
            return Tensor(self.data / other.data)
        return Tensor(self.data / other)
    
    def __floordiv__(self, other: Union[int, float, "Tensor"]) -> "Tensor":
        if isinstance(other, Tensor):
            assert self.could_be_broadcasted(other)
            return Tensor(self.data // other.data)
        return Tensor(self.data // other)
    
    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self * other

    def exp(self):
        return Tensor(np.exp(self.data))

    def dot(self, other):
        if isinstance(other, Tensor):
            assert self.shape()[:-2] == self.shape()[:-2]
            assert self.shape()[-1] == other.shape()[-2]
            return Tensor(np.dot(self.data, other.data))
        return Tensor(np.dot(self.data, other))

    def shape(self):
        return self.data.shape

    def argmax(self, dim=None):
        return np.argmax(self.data, axis=dim)

    def max(self, dim=None):
        return Tensor(np.max(self.data, axis=dim))

    def reshape(self, *args, **kwargs):
        self.data = np.reshape(self.data, *args, **kwargs)
        return self

    def backward(self):
        for value in self.data.flatten():
            value.backward()

    def parameters(self):
        return list(self.data.flatten())

    def __repr__(self):
        return "Tensor\n" + str(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def item(self):
        return self.data.flatten()[0].data

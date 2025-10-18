import math
class Value:
  def __init__(self,data,_children=(),_op="",label=""):
    self.data = data
    self.grad = 0.0
    self._prev = set(_children)
    self._backward = lambda: None
    self._op = _op
    self.label = label



  def __repr__(self):
    '''used to format printing'''
    return f"Value(data={self.data})"

  def __add__(self,other):
    other = other if isinstance(other,Value) else Value(other)

    out = Value(self.data + other.data, (self, other), '+')
    def _backward():
      self.grad += 1.0*out.grad
      other.grad += 1.0*out.grad
    out._backward = _backward
    return out

  def __sub__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data - other.data, (self, other), '-')
    def _backward():
        self.grad += 1.0 * out.grad
        other.grad += -1.0 * out.grad
    out._backward = _backward
    return out

  def __rsub__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    return other - self


  def __mul__(self,other):
    other = other if isinstance(other,Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')
    def _backward():
      self.grad += other.data*out.grad
      other.grad += self.data*out.grad
    out._backward = _backward
    return out

  def __rmul__(self,other):
    # Explicitly define __rmul__
    return self.__mul__(other)


  def __truediv__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data / other.data, (self, other), '/')
    def _backward():
        self.grad += (1 / other.data) * out.grad
        other.grad += (-self.data / (other.data ** 2)) * out.grad
    out._backward = _backward
    return out


  def __pow__(self, power):

        assert isinstance(power, (int, float)), "Power must be numeric"
        out = Value(self.data ** power, (self,), f"**{power}")
        def _backward():
            self.grad += (power * (self.data ** (power - 1))) * out.grad
        out._backward = _backward
        return out

  def tanh(self):
    x = self.data
    t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
    out = Value(t, (self,), 'tanh')
    def _backward():
      self.grad += (1 - t**2) * out.grad
    out._backward = _backward
    return out

  def backward(self):
      topo = []
      visited = set()
      def build_topo(v):
          if v not in visited:
              visited.add(v)
              for child in v._prev:
                  build_topo(child)
              topo.append(v)

      build_topo(self)

      self.grad = 1.0
      for node in reversed(topo):
          node._backward()


from graphviz import Digraph
import random

def trace(root):
  nodes,edges = set(),set()
  def build(v):
    if isinstance(v, Value) and v not in nodes:
      nodes.add(v)
      for child in v._prev:
        edges.add((child,v))
        build(child)
  build(root)
  return nodes,edges

def draw_dot(root):
  dot = Digraph(format='svg',graph_attr={'rankdir':'LR'})

  nodes , edges = trace(root)
  for n in nodes:
    uid = str(id(n))
    dot.node(name=uid,label="{ %s | data %.4f | grad %.4f}" % (n.label, n.data, n.grad),shape='record')
    if n._op:
      # Create a node for the operation
      dot.node(name=uid+n._op, label=n._op)
      # Draw edge from the operation node to the parent node (n)
      dot.edge(uid+n._op, uid)

  for n1,n2 in edges:
    # Draw edge from the child node (n1) to the operation node of the parent (n2)
    dot.edge(str(id(n1)), str(id(n2)) + n2._op)

  return dot

# Check if root is defined and is a Value object before calling draw_dot
if 'root' in globals() and isinstance(root, Value):
    dot = draw_dot(root)
    display(dot)


class Neuron:
  def __init__(self,n_inp):
    self.weights = [Value(random.uniform(-1,1)) for i in range(n_inp)]
    self.bias = Value(random.uniform(-1,1))

  def __call__(self,x):
    s = sum((wi*xi for wi,xi in zip(self.weights,x)), self.bias)
    output = s.tanh()
    return output

  def parameters(self):
    return self.weights + [self.bias]


class Layer:
  def __init__(self,n_inp,n_out):
    self.neurons = [Neuron(n_inp) for i in range(n_out)]

  def __call__(self,x):
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs)==1 else outs

  def parameters(self):
    params = []
    for neuron in self.neurons:
      params.extend(neuron.parameters())
    return params

class MLP:

  def __init__(self,n_inp,n_out): # Added n_inp and n_out as parameters
    #here n_out is a list of outputs
    '''layered_list is consisting of sizes of all layers in mlp'''
    layered_list = [n_inp] + n_out #adding 2 lists
    self.layers = [Layer(layered_list[i],layered_list[i+1]) for i in range(len(n_out))] # creating a layer each time , no of input followed by no of output

  def __call__(self,x):
    for layer in self.layers:
      x = layer(x)
    return x

  def parameters(self):
    params = []
    for layer in self.layers:
      params.extend(layer.parameters())
    return params


def gradient_descent(n,epochs):
  for i in range(epochs):
    yp = [n(x) for x in ds]
    loss = [(y-yp)**2 for y,yp in zip(y,yp)]
    tloss = sum(loss,Value(0))
    for p in n.parameters():
      p.grad = 0

    tloss.backward()

    for p in n.parameters():
      p.data += -0.1*p.grad

  print(tloss)


''' test code '''

x = [1,2,3]
n = MLP(3,[4,3,2,1])
n(x)

t = n(x).backward()
draw_dot(n(x))

ds = [[1,2,3],[2,3,4],[3,4,5]]
y = [0,1,2]


yp = []
for x in ds:
    x_values = []
    for v in x:
        x_values.append(Value(v))
    y_pred = n(x_values)
    yp.append(y_pred)

loss = []
for y_true, y_pred in zip(y, yp):
    loss_value = (Value(y_true) - y_pred) ** 2
    loss.append(loss_value)

tloss = Value(0)
for l in loss:
    tloss += l

print(loss)
tloss = sum(loss,Value(0))
print(tloss)
draw_dot(tloss)
tloss.backward()
print(tloss)
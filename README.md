# Octopy: Fast Tensor and Machine Learning Library

Octopy is a tensor and machine learning library written in C and
wrapped in Python with speed as a fundamental
consideration.

An example of the tensor api is as follows
```python
import octopy_core.tensor as ot

t = ot.Tensor([[1.0, 2.0, 3.0],
    	       [1.0, 2.0, 3.0]])
print(t.shape) # (2, 3)

z = ot.zeros((4, 3))
o = ot.ones((4, 3))

zo = z + o # add two tensors

# matrix multiplication
A = ot.Tensor([[2, 1.5], [2, -1]])
A2 = A @ A
A2._print()
# [[7.00, 1.50],
#  [2.00  4.00]]
```

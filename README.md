# Octopy: Fast Tensor and Machine Learning Library

Octopy is a tensor and machine learning library written in C and
wrapped in Python with speed as a fundamental
consideration.

## Installation
Once you download the repository you can build and install the python module on your local machine by running
```
$ make module
```

## API Example
The following is a brief introduction to the tensor API
```python
# Load the octopy tensor library
import octopy_core.tensor as ot

# Create a tensor from a list
T = ot.Tensor([[1.0, 1.1, 2.4],
               [2.2, 1.9, 8.7],
               [1.1, 0.1, 0.2]])
print(T)
# [[1.000, 2.200, 1.100],
#  [1.100, 1.900, 0.100],
#  [2.400, 8.700, 0.200]]

# Create a tensor of zeros from a shape
T = ot.zeros((4, 3, 2))
print(T)
# [[[0.000, 0.000],
#   [0.000, 0.000],
#   [0.000, 0.000]],

#  [[0.000, 0.000],
#   [0.000, 0.000],
#   [0.000, 0.000]],

#  [[0.000, 0.000],
#   [0.000, 0.000],
#   [0.000, 0.000]],

#  [[0.000, 0.000],
#   [0.000, 0.000],
#   [0.000, 0.000]]]

# Or a tensor of ones
T = ot.ones((3, 10))
print(T)
# [[1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000],
#  [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000],
#  [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000]]

# Or a tensor of random numbers from a uniform distribution
T = ot.rand((2, 3, 5))
print(T)
# [[[0.860, 0.984, 0.852, 0.661, 0.497],
#   [0.199, 0.303, 0.804, 0.513, 0.084],
#   [0.318, 0.262, 0.578, 0.666, 0.724]],

#  [[0.782, 0.318, 0.262, 0.578, 0.666],
#   [0.984, 0.852, 0.661, 0.497, 0.974],
#   [0.303, 0.804, 0.513, 0.084, 0.238]]]

# Add two tensors
T = ot.rand((3, 2))
S = ot.rand((3, 2))
print(T + S)
# [[0.680, 0.857],
#  [1.130, 1.272],
#  [0.632, 1.576]]

# Multiply two tensors
T = ot.rand((4, 2, 10))
S = ot.rand((10, 1, 4))
TS = T @ S
print( TS.shape )
# (4, 2, 1, 4)
print(TS)
# [[[[2.367, 2.036, 2.553, 2.141]],
#   [[2.119, 1.932, 2.438, 1.973]]],

#  [[[2.036, 2.553, 2.141, 2.119]],
#   [[1.932, 2.438, 1.973, 2.728]]],

#  [[[2.553, 2.141, 2.119, 1.932]],
#   [[2.438, 1.973, 2.728, 2.057]]],

#  [[[2.141, 2.119, 1.932, 2.438]],
#   [[1.973, 2.728, 2.057, 2.938]]]]
```

## Notes
Currently random numbers are generated using Evan Sultanik's [Mersenne Twister](https://github.com/ESultanik/mtwister/tree/5b9ae08dda908d800259abc722a31e3717821422) implementation as the base RNG.
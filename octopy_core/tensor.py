from _octopy import _Tensor

def recursive_check_valid (data):
    """
    Recursivly check that the supplied list isn't ragged.

    Call on root.
    """
    leaf_nodes = None

    # Check that all of the entries are the same type
    for ii, dd in enumerate(data):
        if isinstance(dd, list):
            if leaf_nodes != None and leaf_nodes:
                return False
            elif leaf_nodes == None:
                leaf_nodes = False

            if recursive_check_valid(dd) is False:
                return False
        else:
            if leaf_nodes != None and not leaf_nodes:
                return False
            elif leaf_nodes == None:
                leaf_nodes = True

    # check that all of the lists are the same length
    if not leaf_nodes:
        data_len = len(data[0])
        for ii in range(len(data)):
            if len(data[ii]) != data_len:
                return False

    return True

def unroll_nested_list (data, unrolled):
    """
    Flatten a nested list into a single list
    """
    if not isinstance(data[0], list):
        for dd in data:
            unrolled.append(dd)
    else:
        for dd in data:
            unrolled = unroll_nested_list(dd, unrolled)
    
    return unrolled

class Tensor(_Tensor):
    def __init__(self, data=None):
        self.initialized = False
        
        if data is None:
            # Don't initialize the tensor
            return
        
        if type(data) is list:
            self._build_tensor(data)

    def _as_empty(self, ndim: int, dims: tuple):
        super(Tensor, self).__init__(ndim, dims)
        self.initialized = True
        return

    def _build_tensor(self, data):
        if not recursive_check_valid(data):
            raise Exception(
                'Cannot create Tensor from ragged array'
            )
        
        axes = [len(data)]
        child = data[0]
        
        while isinstance(child, list):
            axes.append(len(child))
            child = child[0]
        
        super(Tensor, self).__init__(len(axes), tuple(axes))
        self._assign_data_from_list(
            unroll_nested_list(data, [])
        )

        self.initialize = True
        return

def ones(shape):
    if isinstance(shape, list):
        shape = tuple(shape)
    elif not isinstance(shape, tuple):
        raise Exception('Shape must be list or tuple.')

    T = Tensor()
    T._as_empty(len(shape), shape)
    T._to_ones()
    return T

def zeros(shape):
    if isinstance(shape, list):
        shape = tuple(shape)
    elif not isinstance(shape, tuple):
        raise Exception('Shape must be list or tuple.')

    T = Tensor()
    T._as_empty(len(shape), shape)
    
    return T

if __name__ == "__main__":
    # print(recursive_check_valid([1, 2, 3])) # True
    # print(recursive_check_valid([[1, 2, 3], [2,3, 4]])) # True
    # print(recursive_check_valid([[1, 2, 3], [2, 4]])) # False
    # print(recursive_check_valid([[1, 2, 3], 2, 4, 4])) # False


    # TODO: Do tests for making sure that the bad argument handling
    # works

    t = tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    print(t.shape)

    t = tensor([[[1, 1], [1, 1], [1, 1]], [[1, 1], [1, 1], [1, 1]]])
    print(t.shape)

    t = ones((3, 5, 6, 10, 28, 10, 34, 5))
    print(t.shape)

    t = zeros((64, 28, 28, 3))
    print(t.shape)

    print(t[1, 2, 1, 1])
    t[0, 0, 0, 0] = 1
    print(t[0, 0, 0, 0])

    t = ones((2, 2))
    t[0, 0] = 1
    t[0, 1] = 2
    t[1, 0] = 3
    t[1, 1] = 4

    print(t[1, 0])

    t = tensor([[1, 2], [3, 4]])
    print(t)

    

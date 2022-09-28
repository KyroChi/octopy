from octopy import _Tensor

class opTensor (_Tensor):
    """
    Python wrapper for the _Tensor class.

    _Tensor is written in C. The opTensor wrapper allows us to write
    non-preformance critical methods in Python.

    Indexing is done through Python and is slow. Performance critical
    operations should call the superclass' _ methods directly to avoid
    slice parsing.

    Tensor math operations and chained tensor operations are
    implemented directly in the C API.

    Performance crtical operations are inherited by the superclass
    _Tensor.

    attributes
    ==========
    shape: inherited from parent _Tensor class
    """
    def __init__(self, shape):
        if isinstance(shape, list):
            shape = tuple(shape)

        if not isinstance(shape, list) and \
           not isinstance(shape, tuple):
            raise Exception('shape must be tuple or list')
            
        super(opTensor, self).__init__(len(shape), shape)

    def _assign_data_from_list(self, data):
        """
        Assumes that the data has been checked.
        """
        # BROKEN: This indexes row first and I need to index column
        # first. This transposes the first rows?

        # The problem is that the data will be row first then column,
        # but stored as column first.
        data = unroll_nested_list(data, [])
        
        for ii, dd in enumerate(data):
            self._set_tensor_linear(ii, dd)

    def __getitem__(self, i):
        # No slices yet
        if not isinstance(i, tuple):
            raise Exception('Currently only supports tuple slices')

        return self._get_tensor(i)

    def __setitem__(self, i, v):
        # No slices yet
        if not isinstance(i, tuple):
            raise Exception('Currently only supports tuple slices')

        if isinstance(v, int):
            v = float(v)

        if not isinstance(v, float):
            raise Exception(
                'Tensor cannot accept non-floating point type %s'
                % str(type(v))
            )

        self._set_tensor(i, v)

    def __str__(self):
        return "opTensor(shape=" + str(self.shape) + ")"

    def print_values(self):
        """
        Use for debugging. Pretty prints the tensor
        """
        data = self._dump()


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

def tensor(data, as_column_first=True):
    """ Create an opTensor object from a nested array
    """
    if len(data) == 0:
        raise Exception(
            'Cannot create tensor from empty array'
        )

    # check the validity of the array
    if not recursive_check_valid(data):
        raise Exception(
            'Cannot create opTensor from ragged array'
        )

    # must convert data from row first to column first.
    if as_column_first:
        pass

    axes = [len(data)]
    child = data[0]
    
    while isinstance(child, list):
        axes.append(len(child))
        child = child[0]

    T = _Tensor(len(axes), tuple(axes))
    T._assign_data_from_list(unroll_nested_list(data, []))
    return T

def zeros(shape):
    if isinstance(shape, list):
        shape = tuple(shape)
    elif not isinstance(shape, tuple):
        raise Exception('Shape must be list or tuple.')
    
    return _Tensor(len(shape), shape)

def ones(shape):
    if isinstance(shape, list):
        shape = tuple(shape)
    elif not isinstance(shape, tuple):
        raise Exception('Shape must be list or tuple.')
    
    T = _Tensor(len(shape), shape)
    T._to_ones()
    return T

if __name__ == "__main__":
    # print(recursive_check_valid([1, 2, 3])) # True
    # print(recursive_check_valid([[1, 2, 3], [2,3, 4]])) # True
    # print(recursive_check_valid([[1, 2, 3], [2, 4]])) # False
    # print(recursive_check_valid([[1, 2, 3], 2, 4, 4])) # False

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

    

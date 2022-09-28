#ifndef PY_TENSOR_H
#define PY_TENSOR_H

#include <Python.h>
#include "structmember.h"

#include "../math/tensor.h"

static PyTypeObject PyTensorType;

typedef struct{
	PyObject_HEAD        // Semicolon included in header
	PyObject *shape;      // axes as a tuple
	/* _tensor holds unsigned int rank and unsigned int *axes */
	Tensor *_tensor;      	
} PyTensor;

PyTensor *new_PyTensor_from_tensor (Tensor *);
unsigned int *get_idxs_from_PyTuple (PyObject *);
PyObject *get_PyTuple_from_idxs (unsigned int, unsigned int *);
int check_tuple_size (PyTensor *T, PyObject *py_tuple);

void PyTensor_dealloc (PyTensor *);
PyObject *PyTensor_new (PyTypeObject *, PyObject *, PyObject *);
int PyTensor_init (PyTensor *, PyObject *, PyObject *);

PyObject *PyTensor_set_tensor_linear (PyTensor *, PyObject *);
PyObject *PyTensor_get_tensor (PyTensor *, PyObject *, PyObject *);
PyObject *PyTensor_set_tensor (PyTensor *, PyObject *, PyObject *);

PyObject *PyTensor_to_ones (PyTensor *);

PyObject *PyTensor_add_tensor (PyTensor *, PyObject *);
PyObject *PyTensor_assign_data_from_list (PyTensor *, PyObject *);
PyObject *PyTensor_matmul (PyTensor *, PyObject *);
PyObject *PyTensor_dump (PyTensor *);

#endif

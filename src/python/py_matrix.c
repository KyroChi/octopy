#include <Python.h>
#include "structmember.h"

#include "../math/tensor.h"
#include "../math/matrix.h"

void
PyMatrix_dealloc (PyMatrix *self)
{
	free_matrix(self->_matrix);
	Py_XDECREF(self->shape);
	Py_TYPE(self)->tp_free( (PyObject *) self );

	return;
}

PyObject *
PyMatrix_new (PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	PyMatrix *self = NULL;

	self = (PyMatrix *) type->tp_alloc(type, 0);

	if (self == NULL) {
		return NULL;
	}

	self->shape = NULL;
	self->_matrix = NULL;

	return (PyMatrix *) self;
}

int
PyTensor_init (PyTensor *self, PyObject *args, PyObject *kwds)
{
	unsigned int n_rows, n_cols;
	PyObject *py_shape, *tmp;

	if ( !PyArg_ParseTuple(args, "O", &py_shape) ) {
		// TODO: Better error parsing
		PyErr_SetString(PyExc_TypeError,
				"Must be called with O type");
		return -1;
	}

	if ( PyTyple_Size(py_shape) != 2 ) {
		PyErr_SetString(PyExc_Exception,
				"Shape must be rank 2");
		return -1;
	}

	if ( py_shape ) {
		tmp = self->shape;
		Py_INCREF(py_shape);
		self->shape = py_shape;
		Py_XDECREF(tmp);
	}

	n_rows = (unsigned int)
		PyLong_AsUnsignedLong(PyTuple_GetItem(py_shape, 0));
	n_cols = (unsigned int)
		PyLong_AsUnsignedLong(PyTuple_GetItem(py_shape, 1));

	self->_matrix = new_matrix(n_rows, n_cols);

	return 0;
}

PyMatrix *
new_PyMatrix_from_matrix (Matrix *A)
{
	PyMatrix *mat;
	PyObject *args;
	unsigned int n_rows, n_cols;

	mat = (PyMatrix *) PyMatrix_new(&PyMatrixType, NULL, NULL);

	args = PyTuple_New(2);
	PyTuple_SetItem(args,
			0,
			PyLong_FromUnsignedLong( (unsigned long) n_rows ));
	PyTuple_SetItem(args,
			1,
			PyLong_FromUnsignedLong( (unsigned long) n_cols ));

	PyMatrix_init(mat, args, NULL);
	mat->_matrix = A;

	return mat;
}

PyObject *
PyMatrix_matmul (PyMatrix *self, PyObject *args)
{
	PyMatrix *B;
	Matrix *prod;

	if ( !PyArg_ParseTuple(args, "O", &b) ) {
		// TODO: Set error flags
		return NULL;
	}

	prod = matrix_matmul(self->_matrix, B->_matrix);
	Py_XDECREF(B);

	return (PyObject *) new_PyMatrix_from_matrix(prod);
}

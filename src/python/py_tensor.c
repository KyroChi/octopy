/**
 * Wrapper class for the C Tensor struct
 */

#include <stdio.h>

#include <Python.h>
#include "structmember.h"

#include "../math/tensor.h"
#include "../math/initializers.h"
#include "py_tensor.h"

unsigned int *
get_idxs_from_PyTuple (PyObject *tuple)
/* You should know how long this tuple is prior to obtaining the
 * inexes */
{
	unsigned int length = (unsigned int) PyTuple_Size(tuple);
	unsigned int *idxs = malloc( sizeof(unsigned int) * length);

	unsigned int ii;
	for (ii = 0; ii < length; ii += 1) {
		// TODO: This call looks like shit.
		idxs[ii] = (unsigned int)
			PyLong_AsLong(
			         PyTuple_GetItem(tuple, ii)
					      );
	}

	return idxs;
}

PyObject *
get_PyTuple_from_idxs (unsigned int rank, unsigned int *shape)
{
	PyObject *py_tuple = PyTuple_New(rank);
	
	if ( !py_tuple ) {
		// Set error flags
		return NULL;
	}

	unsigned int ii;
	for (ii = 0; ii < rank; ii += 1) {
		PyTuple_SetItem(py_tuple, ii,
				PyLong_FromLong(shape[ii]));
	}

	return py_tuple;
}

int
check_tuple_size (PyTensor *T, PyObject *py_tuple)
{
	if ( !((unsigned int) PyTuple_Size(py_tuple) ==
	       T->_tensor->rank) ) {
		PyErr_SetString(PyExc_IndexError,
				"attempted to index an array using a tuple of incorrect shape");
		return -1;
	}

	return 0;
}

void
PyTensor_dealloc (PyTensor *self)
{
	free_tensor(self->_tensor);
	Py_XDECREF(self->shape);
	Py_TYPE(self)->tp_free( (PyObject*) self );

	return;
}

PyObject *
PyTensor_new (PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	PyTensor *self = NULL;

	self = (PyTensor *) type->tp_alloc(type, 0);

	if (self != NULL) {
		self->shape = Py_None;
		if (self->shape == NULL) {
			Py_DECREF(self);
			return NULL;
		}

		self->_tensor = NULL;
	}	

	return (PyObject *) self;
}

int
PyTensor_init (PyTensor *self, PyObject *args, PyObject *kwds)
{
	unsigned int rank;
	unsigned int *shape;
	
	PyObject *py_tuple, *tmp;

	if ( !PyArg_ParseTuple(args, "iO", &rank, &py_tuple) ) {
		// TODO: Better error parsing
		PyErr_SetString(PyExc_TypeError,
				"Must be called with iO type");
		return -1;
	}

	// Check the type of py_tuple

	unsigned int length = (unsigned int) PyTuple_Size(py_tuple);
	if ( length != rank ) {
		PyErr_SetString(PyExc_Exception,
				"Rank must match length of axes");
		return -1;
	}

	if ( py_tuple ) {
		tmp = self->shape;
		Py_INCREF(py_tuple);
		self->shape = py_tuple;
		Py_XDECREF(tmp);
	}

	shape = malloc( sizeof(unsigned int) * rank );
	
	unsigned int ii;
	unsigned int num_entries = 1;
	for (ii = 0; ii < rank; ii += 1) {
		// TODO: This call looks like shit.
		shape[ii] = (unsigned int)
			PyLong_AsLong(
			         PyTuple_GetItem(py_tuple, ii)
					      );
		num_entries *= shape[ii];
	}

	self->_tensor = new_tensor(rank, shape);
	self->_tensor->size = num_entries;

	return 0;
}

PyObject *
PyTensor_set_tensor_linear (PyTensor *self, PyObject *args)
{
	unsigned int index;
	float v;
	
	if ( !PyArg_ParseTuple(args, "if", &index, &v) ) {
		// TODO: Set error flags. Failed parse.
		PyErr_SetString(PyExc_Exception,
				"Parse failure");
		return NULL;
	}

	if ( self->_tensor->size < index ) {
		PyErr_SetString(PyExc_IndexError,
				"index out of bounds");
		return NULL;
	}

	self->_tensor->data[index] = v;

	return Py_None;
}

PyObject *
PyTensor_get_tensor (PyTensor *self, PyObject *py_tuple)
{
	unsigned int *idxs;
	float v;

	// TODO: Python argument handling

        if ( 0 > check_tuple_size(self, py_tuple) ) {
		// Error set in call
		return NULL;
	}

	idxs = get_idxs_from_PyTuple(py_tuple);

	if ( 0 > check_index_validity(self->_tensor, idxs) ) {
		PyErr_SetString(PyExc_IndexError,
				"index out of bound for tensor");
		return NULL;
	}
	
	v = get_tensor(self->_tensor, idxs);
	
	free(idxs);
	Py_XDECREF(py_tuple);
	
	return Py_BuildValue("f", v);
}

int
PyTensor_set_tensor (PyTensor *self, PyObject *ind_tuple,
		     PyObject *value)
{
	unsigned int *idxs;
	float v;

	v = (float) PyFloat_AsDouble(value);

	if ( 0 > check_tuple_size(self, ind_tuple) ) {
		// Error set in call
		return -1;
	}

	idxs = get_idxs_from_PyTuple(ind_tuple);

	if ( 0 > check_index_validity(self->_tensor, idxs) ) {
		PyErr_SetString(PyExc_IndexError,
				"index out of bound for tensor");
		return -1;
	}
	
	set_tensor(self->_tensor, idxs, v);
	
	free(idxs);
	/* Py_XDECREF(py_tuple); */
	
	return 0;
}

PyObject *
PyTensor_to_ones (PyTensor *self)
{
	to_ones(self->_tensor);
	return Py_None;
}

PyObject *
PyTensor_to_rand (PyTensor *self)
{
	initialize_tensor(self->_tensor,
			  &initializer_default_uniform);
	return Py_None;
}

PyObject *
PyTensor_add_tensor (PyObject *s, PyObject *w)
{
	Tensor *sum;
	sum = tensor_add_s(((PyTensor *) s)->_tensor,
			   ((PyTensor *) w)->_tensor);

	return (PyObject *) new_PyTensor_from_tensor(sum);
}

PyObject *
PyTensor_scalar_mult (PyObject *self, PyObject *po)
/* TODO: Only accepts po as type float. Do type error handling and
 * also accept integers.
 * 
 */
{
	Tensor *T = ((PyTensor *) self)->_tensor;
	float a = PyFloat_AsDouble(po);
		
	return (PyObject *)
		new_PyTensor_from_tensor(scalar_multiply(T, a));
}

PyObject *
PyTensor_mat_mul (PyObject *self, PyObject *T)
{
	Tensor *A = ((PyTensor *) self)->_tensor;
	Tensor *B = ((PyTensor *) T)->_tensor;

	// TODO: Size checking!!

	return (PyObject *)
		new_PyTensor_from_tensor(tensor_matmul(A, B));
}

PyObject *
PyTensor_assign_data_from_list (PyTensor *self, PyObject *args)
/**
 * Assigns data from an unrolled list to a tensor. The parsing and
 * unrolling is done by the caller (probably in Python).
 *
 * TODO: List is indexed wrong, must swap the first two indicies.
 */
{
	PyObject *data;
	unsigned int size, ii;
	float v;

	if ( !PyArg_ParseTuple(args, "O", &data) ) {
		// TODO: Set error flags
		return NULL;
	}

	size = PyList_Size(data);

	for (ii = 0; ii < size; ii += 1) {
		v = (float) PyFloat_AsDouble(PyList_GetItem(data, ii));
		
		if ( PyErr_Occurred() ) {
			PyErr_SetString(PyExc_Exception,
				"Failed to convert data to float?");
			return NULL;
		}
		
		self->_tensor->data[ii] = v;
	}

	Py_XDECREF(data);

	return Py_None;
}

PyObject *
PyTensor_matmul (PyTensor *self, PyObject *args)
{
	PyTensor *v;
	Tensor *prod;

	if ( !PyArg_ParseTuple(args, "O", &v) ) {
		// TODO: Set error flags
		return NULL;
	}

	prod = tensor_matmul(self->_tensor, v->_tensor);
	Py_XDECREF(v);

	return (PyObject *) new_PyTensor_from_tensor(prod);
}

PyObject *
PyTensor_dump (PyTensor *self)
/**
 * Return the self->_tensor->data array as a Python array.
 *
 * Must be re-nested by caller.
 */
{
	return NULL;
}

PyObject *
PyTensor_print (PyTensor *self)
{
	tensor_print(self->_tensor);
	return Py_None;
}

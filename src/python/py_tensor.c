/**
 * Wrapper class for the underlying Tensor struct
 */

#include <Python.h>
#include "structmember.h"

#include "../math/tensor.h"

typedef struct{
	PyObject_HEAD        // Semicolon included in header
	PyObject *shape;      // axes as a tuple
	/* _tensor holds unsigned int rank and unsigned int *axes */
	Tensor *_tensor;      	
} PyTensor;

static void
PyTensor_dealloc (PyTensor* self)
{
	free_tensor(self->_tensor);
	Py_XDECREF(self->shape);
	Py_TYPE(self)->tp_free( (PyObject*) self );

	return;
}

static PyObject *
PyTensor_new (PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	PyTensor *self = NULL;
		
	self = (PyTensor *)type->tp_alloc(type, 0);

	if (self == NULL) {
		return NULL;
	}

	self->shape = NULL;
	self->_tensor = NULL;

	return (PyObject *) self;
}

static unsigned int *
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

static int
PyTensor_init (PyTensor *self, PyObject *args, PyObject *kwds)
{
	unsigned int rank;
	unsigned int *shape;
	
	PyObject *py_tuple, *tmp;

	if ( !PyArg_ParseTuple(args, "iO", &rank, &py_tuple) ) {
		// TODO: set error flags
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

static int
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

static int
PyTensor_set_tensor (PyTensor *self, PyObject *args)
{
	PyObject *py_tuple;
	unsigned int *idxs;
	float v;

	if ( !PyArg_ParseTuple(args, "Of", &py_tuple, &v) ) {
		// TODO: Set error flags. Failed parse.
		return -1;
	}

	if ( 0 > check_tuple_size(self, py_tuple) ) {
		// Error set in call
		return -1;
	}

	idxs = get_idxs_from_PyTuple(py_tuple);

	if ( 0 > check_index_validity(self->_tensor, idxs) ) {
		PyErr_SetString(PyExc_IndexError,
				"index out of bound for tensor");
		return -1;
	}
	
	set_tensor(self->_tensor, idxs, v);
	
	free(idxs);
	Py_XDECREF(py_tuple);
	
	return 0;
}

PyObject *
PyTensor_get_tensor (PyTensor *self, PyObject *args)
{
	PyObject *py_tuple;
	unsigned int *idxs;
	float v;

	if ( !PyArg_ParseTuple(args, "O", &py_tuple) ) {
		// TODO: Set error flags. Failed Parse.
		return NULL;
	}

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

void
PyTensor_to_ones (PyTensor *self)
{
	to_ones(self->_tensor);
	return;
}

static PyMemberDef PyTensor_members[] = {
	{"shape", T_OBJECT_EX, offsetof(PyTensor, shape), 0,
	 "tuple containing size of axes"},
	{NULL} /* Sentinal */
};

static PyMethodDef PyTensor_methods[] = {
	{"_set_tensor", (PyCFunction) PyTensor_set_tensor,
	 METH_VARARGS,
	 "Set the value at supplied axes."},
	{"_get_tensor", (PyCFunction) PyTensor_get_tensor,
	 METH_VARARGS,
	 "Get the value at supplied axes."},
	{"_to_ones", (PyCFunction) PyTensor_to_ones,
	 METH_VARARGS,
	 "Set all entries to 1."},
	{NULL}
};

static PyTypeObject PyTensorType = {
	PyVarObject_HEAD_INIT(NULL, 0)
	"octopy._Tensor",             /* tp_name */
	sizeof(PyTensor),             /* tp_basicsize */
	0,                         /* tp_itemsize */
	(destructor)PyTensor_dealloc, /* tp_dealloc */
	0,                         /* tp_print */
	0,                         /* tp_getattr */
	0,                         /* tp_setattr */
	0,                         /* tp_reserved */
	0,                         /* tp_repr */
	0,                         /* tp_as_number */
	0,                         /* tp_as_sequence */
	0,                         /* tp_as_mapping */
	0,                         /* tp_hash  */
	0,                         /* tp_call */
	0,                         /* tp_str */
	0,                         /* tp_getattro */
	0,                         /* tp_setattro */
	0,                         /* tp_as_buffer */
	Py_TPFLAGS_DEFAULT |
        Py_TPFLAGS_BASETYPE,   /* tp_flags */
	"Tensor object",           /* tp_doc */
	0,                         /* tp_traverse */
	0,                         /* tp_clear */
	0,                         /* tp_richcompare */
	0,                         /* tp_weaklistoffset */
	0,                         /* tp_iter */
	0,                         /* tp_iternext */
	PyTensor_methods,             /* tp_methods */  /* Type methods */
	PyTensor_members,             /* tp_members */
	0,                         /* tp_getset */
	0,                         /* tp_base */
	0,                         /* tp_dict */
	0,                         /* tp_descr_get */
	0,                         /* tp_descr_set */
	0,                         /* tp_dictoffset */
	(initproc)PyTensor_init,      /* tp_init */
	0,                         /* tp_alloc */
	PyTensor_new,                 /* tp_new */
};

static PyModuleDef octopymodule = {
	PyModuleDef_HEAD_INIT,
	"octopy",
	"Tensor and machine learning library.",
	-1,
	NULL, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC
PyInit_octopy(void)
{
	PyObject* m;

	if (PyType_Ready(&PyTensorType) < 0)
		return NULL;

	m = PyModule_Create(&octopymodule);
	if (m == NULL)
		return NULL;

	Py_INCREF(&PyTensorType);
	PyModule_AddObject(m, "_Tensor", (PyObject *)&PyTensorType);
	return m;
}

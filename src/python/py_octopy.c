#include <Python.h>
#include "py_tensor.h"
#include "py_octopy.h"

static PyMemberDef PyTensor_members[] = {
	{"shape", T_OBJECT_EX, offsetof(PyTensor, shape), 0,
	 "tuple containing size of axes"},
	{NULL} /* Sentinal */
};

static PyMethodDef PyTensor_methods[] = {
	{"_set_tensor",
	 (PyCFunction) PyTensor_set_tensor,
	 METH_VARARGS,
	 "Set the value at supplied axes."
	},
	{"_set_tensor_linear",
	 (PyCFunction) PyTensor_set_tensor_linear,
	 METH_VARARGS,
	 "Set the value at a point in the raw data array."
	},
	{"_to_ones",
	 (PyCFunction) PyTensor_to_ones,
	 METH_VARARGS,
	 "Set all entries to 1."
	},
	{"_dump",
	 (PyCFunction) PyTensor_dump,
	 METH_NOARGS,
	 "Dump Tensor data to an array"
	},
	{"_assign_data_from_list",
	 (PyCFunction) PyTensor_assign_data_from_list,
	 METH_VARARGS,
	 "assign data from an unrolled list"
	},
	{NULL} /* Sentinal */
};

static PyMappingMethods PyTensorMappingMethods = {
	.mp_subscript = (binaryfunc) PyTensor_get_tensor,
	.mp_ass_subscript = (objobjargproc) PyTensor_set_tensor,
};

static PyNumberMethods PyTensorNumberMethods = {
	.nb_add = (binaryfunc) PyTensor_add_tensor,
};

static PyTypeObject PyTensorType = {
	PyVarObject_HEAD_INIT(NULL, 0)
	.tp_name = "_octopy._Tensor",
	.tp_basicsize = sizeof(PyTensor),
	.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
	.tp_doc = "Tensor object",
	.tp_new = PyTensor_new,
	.tp_init = (initproc)PyTensor_init,
	.tp_dealloc = (destructor)PyTensor_dealloc,
	.tp_methods = PyTensor_methods,
	.tp_members = PyTensor_members,
	.tp_as_mapping = &PyTensorMappingMethods,
	.tp_as_number = &PyTensorNumberMethods,
};

static PyModuleDef octopymodule = {
	PyModuleDef_HEAD_INIT,
	"_octopy",
	"Tensor and machine learning library.",
	-1,
	NULL, NULL, NULL, NULL, NULL
};

PyTensor *
new_PyTensor_from_tensor (Tensor *T)
{
	PyTensor *res = NULL;
	PyObject *py_rank, *py_shape, *args;

	res = (PyTensor *) PyTensor_new(&PyTensorType, NULL, NULL);
	
	py_rank = Py_BuildValue("i", (int) T->rank);
	py_shape = get_PyTuple_from_idxs(T->rank, T->shape);

	args = PyTuple_New(2);
	PyTuple_SetItem(args, 0, py_rank);
	PyTuple_SetItem(args, 1, py_shape);

	PyTensor_init(res, args, NULL);
	res->_tensor = T;

	Py_XDECREF(py_rank);
	Py_XDECREF(py_shape);
	Py_XDECREF(args);

	// TODO: Free memory

	return res;
}

PyMODINIT_FUNC
PyInit__octopy(void)
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

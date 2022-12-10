#ifndef PY_MATRIX_H
#define PY_MATRIX_H

#include <Python.h>
#include "structmember.h"

#include "../math/tensor.h"
#include "../math/matrix.h"

typedef struct {
	PyObject_HEAD        // Semicolon included in header
	PyObject *shape;
	Matrix *_matrix;
} PyMatrix;

PyMatrix *new_PyMatrix_from_matrix (Matrix *);

void PyMatrix_dealloc (PyMatrix *);
PyObject *PyMatrix_new (PyTypeObject *, PyObject *, PyObject *);

PyObject *PyMatrix_matmul (PyMatrix *, PyObject *);

#endif

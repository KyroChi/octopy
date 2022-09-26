#include <stdio.h>
#include <stdlib.h>

#include "../octopy_helper.h"
#include "tensor.h"

Tensor *
new_tensor (unsigned int rank, unsigned int *shape)
/* COPIES shape into the tensor. */
{
	Tensor *ptr;
	ptr = malloc( sizeof(*ptr) );
	if (ptr == NULL) {
		// TODO: Set error flags
		return NULL;
	}

	ptr->rank = rank;
	ptr->shape = malloc( sizeof(unsigned int) * rank );
	if (ptr->shape == NULL) {
		// TODO: Set error flags
		return NULL;
	}
	
	array_cpy_uint(shape, ptr->shape, rank);

	unsigned int tensor_size = 1;
	
	unsigned int ii;
	for (ii = 0; ii < rank; ii += 1) {
		tensor_size *= shape[ii];
	}

	ptr->size = tensor_size;

	// Initialize with zero data
	ptr->data = calloc(tensor_size, sizeof(float));
	if (ptr->data == NULL) {
		// TODO: Set error flags
		return NULL;
	}

	return ptr;
}

void
free_tensor (Tensor *T)
{
	free(T->data);
	free(T->shape);
	free(T);

	return;
}

unsigned int
get_linear_index (unsigned int rank, unsigned int *shape,
		  unsigned int *idxs)
{
	unsigned int index = 0;
	
	unsigned int ii;
	for (ii = 0; ii < (rank - 1); ii += 1) {
		index += shape[ii] * idxs[ii];
	}

	index += idxs[ii];

	return index;
}

void
set_tensor(Tensor *T, unsigned int *idxs, float v)
/**
 * Checking of idxs is done by caller to avoid double calling
 */
{
	unsigned int index = get_linear_index(T->rank, T->shape, idxs);
	T->data[index] = v;
	return;
}

float
get_tensor (Tensor *T, unsigned int *idxs)
/* idxs length must match rank */
{
	unsigned int index = get_linear_index(T->rank, T->shape, idxs);
	return T->data[index];
}

int
check_index_validity (Tensor *T, unsigned int *idxs)
/**
 * Assumes that the caller has already checked that the length of idxs 
 * is correct .
 */
{
	unsigned int rank = T->rank;

	unsigned int ii;
	for (ii = 0; ii < rank; ii += 1) {
		if (T->shape[ii] <= idxs[ii]) {
			return -1;
		}
	}

	return 0;
}

float one (float a) { UNUSED(a); return 1.0; }

void
to_ones (Tensor *T)
{
	_tensor_map_subroutine(T, T, &one);
	return;
}

void
_tensor_map_subroutine (Tensor *in, Tensor* out,
			float (*map_fun)(float))
/**
 * Call _matrix_map_subroutine(T, T, &fun) for inplace
 */
{
	// TODO: Check that in and out are same dimension
	// fast for inpace: check that they are same pointer
	float a;

	// This loop avoids having to index into the tensor.
	unsigned int ii;
	for (ii = 0; ii < in->size; ii += 1) {
		a = in->data[ii];
		out->data[ii] = map_fun(a);
	}

	return;
}

Tensor *
tensor_transpose (Tensor *T, unsigned int ax1, unsigned int ax2)
/* Is there a better way to do this? */
{ /*
	// TODO: Check validity of axes
	unsigned int *new_axes = calloc(T->rank,
					sizeof(unsigned int));
	array_cpy_uint(T->axes, new_axes);
	
	unsigned int tmp = 0;
	tmp = new_axes[ax1];
	new_axes[ax1] = new_axes[ax2];
	new_axes[ax2] = tmp;
	
	Tensor *newT = new_tensor(T->rank, new_axes);

	float ltmp;

	
	unsigned int ii, jj;
	for (ii = 0; ii < rank; ii += 1) {
		if (ii == ax1) {
		} else if (ii == ax2) {
			// Do nothing
		} else {
			for (jj = 0; jj < new_axes[ii]; jj += 1) {
				set_tensor(newT, 
			}
		}
		} */ return NULL; // Finish this
}

void
tensor_transpose_inplace (Tensor *T, unsigned int ax1,
			  unsigned int ax2)
{
	return;
}
